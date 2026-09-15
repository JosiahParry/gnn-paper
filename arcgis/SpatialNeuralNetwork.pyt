import os

import arcpy
import numpy as np

NEIGHBORHOOD_TYPES = [
    "K nearest neighbors",
    "Contiguity edges corners",
    "Contiguity edges only",
]

NUMERIC = ["Short", "Long", "Float", "Double"]


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        raise arcpy.ExecuteError(
            "PyTorch is not available in this Python environment. Install the "
            "Deep Learning Essentials package for ArcGIS Pro, or clone the "
            "default environment and install pytorch into the clone."
        )


def _read_table(fc, fields):
    arr = arcpy.da.FeatureClassToNumPyArray(
        fc, ["OID@", "SHAPE@XY"] + list(fields), skip_nulls=False
    )
    if arr.shape[0] == 0:
        raise arcpy.ExecuteError(f"{fc} contains no features.")
    xy = np.vstack([arr["SHAPE@XY"][:, 0], arr["SHAPE@XY"][:, 1]]).T.astype(np.float64)
    if fields:
        cols = np.column_stack([arr[f].astype(np.float64) for f in fields])
        if not np.all(np.isfinite(cols)):
            raise arcpy.ExecuteError(
                f"Null or non-finite values found in {', '.join(fields)}. "
                "Remove or impute them before running this tool."
            )
    else:
        cols = np.empty((arr.shape[0], 0))
    return arr["OID@"], xy, cols


def _knn_edges(xy, k):
    n = xy.shape[0]
    k = int(min(k, n - 1))
    if k < 1:
        raise arcpy.ExecuteError("At least two features are required.")
    try:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(xy).query(xy, k=k + 1)
        nbr = idx[:, 1:]
    except ImportError:
        nbr = np.empty((n, k), dtype=np.int64)
        for i in range(n):
            d = np.sum((xy - xy[i]) ** 2, axis=1)
            d[i] = np.inf
            nbr[i] = np.argpartition(d, k)[:k]
    src = np.repeat(np.arange(n), k)
    return src, nbr.reshape(-1).astype(np.int64)


def _contiguity_edges(fc, oids, both_corners):
    oid_to_row = {int(o): i for i, o in enumerate(oids)}
    tbl = arcpy.CreateUniqueName("nbrs", "in_memory")
    arcpy.analysis.PolygonNeighbors(
        fc, tbl, None, area_overlap="NO_AREA_OVERLAP",
        both_sides="BOTH_SIDES", out_linear_units="METERS",
    )
    oid_field = arcpy.Describe(fc).OIDFieldName
    fields = [f"src_{oid_field}", f"nbr_{oid_field}", "LENGTH"]
    src, dst = [], []
    with arcpy.da.SearchCursor(tbl, fields) as cur:
        for s, d, length in cur:
            if s not in oid_to_row or d not in oid_to_row:
                continue
            # Rook contiguity requires a shared edge, not just a shared vertex.
            if not both_corners and not (length and length > 0):
                continue
            src.append(oid_to_row[s])
            dst.append(oid_to_row[d])
    arcpy.management.Delete(tbl)
    if not src:
        raise arcpy.ExecuteError(
            "No contiguity relationships were found. Check that the input is a "
            "polygon feature class with shared boundaries."
        )
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)


def _edges(fc, oids, xy, nbr_type, k):
    if nbr_type == "K nearest neighbors":
        return _knn_edges(xy, k)
    return _contiguity_edges(fc, oids, nbr_type != "Contiguity edges only")


def _row_standardized_adj(torch, src, dst, n, self_loops=True):
    if self_loops:
        src = np.concatenate([src, np.arange(n)])
        dst = np.concatenate([dst, np.arange(n)])
    counts = np.bincount(src, minlength=n).astype(np.float64)
    counts[counts == 0] = 1.0
    w = 1.0 / counts[src]
    idx = torch.tensor(np.vstack([src, dst]), dtype=torch.int64)
    return torch.sparse_coo_tensor(
        idx, torch.tensor(w, dtype=torch.float32), (n, n)
    ).coalesce()


def _build_model(torch, nn, n_features, hidden, use_layernorm):
    class SageLayer(nn.Module):
        def __init__(self, f_in, f_out):
            super().__init__()
            self.self_lin = nn.Linear(f_in, f_out)
            self.neigh_lin = nn.Linear(f_in, f_out, bias=False)
            self.norm = nn.LayerNorm(f_out) if use_layernorm else None
            self.act = nn.ReLU()

        def forward(self, x, adj):
            h = self.self_lin(x) + self.neigh_lin(torch.sparse.mm(adj, x))
            if self.norm is not None:
                h = self.norm(h)
            return self.act(h)

    class Sage(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [n_features] + list(hidden)
            self.layers = nn.ModuleList(
                SageLayer(dims[i], dims[i + 1]) for i in range(len(hidden))
            )
            self.out = nn.Linear(dims[-1], 1)

        def forward(self, x, adj):
            for layer in self.layers:
                x = layer(x, adj)
            return self.out(x)

    return Sage()


def _warn_geographic(fc, nbr_type):
    if nbr_type == "K nearest neighbors":
        if arcpy.Describe(fc).spatialReference.type == "Geographic":
            arcpy.AddWarning(
                "Input features use a geographic coordinate system, so nearest "
                "neighbors are found using degrees. Project the data for "
                "distances to be meaningful."
            )


def _check_polygon(param_fc, param_nbr):
    if param_nbr.valueAsText and param_nbr.valueAsText.startswith("Contiguity"):
        if param_fc.value:
            if arcpy.Describe(param_fc.value).shapeType != "Polygon":
                param_nbr.setErrorMessage(
                    "Contiguity neighborhoods require polygon features. "
                    "Use K nearest neighbors for points or lines."
                )


def _neighborhood_params(category=None):
    nbr_type = arcpy.Parameter(
        displayName="Neighborhood Type",
        name="neighborhood_type",
        datatype="GPString",
        parameterType="Required",
        direction="Input",
        category=category,
    )
    nbr_type.filter.list = NEIGHBORHOOD_TYPES
    nbr_type.value = NEIGHBORHOOD_TYPES[0]

    n_nbrs = arcpy.Parameter(
        displayName="Number of Neighbors",
        name="number_of_neighbors",
        datatype="GPLong",
        parameterType="Optional",
        direction="Input",
        category=category,
    )
    n_nbrs.value = 15
    n_nbrs.filter.type = "Range"
    n_nbrs.filter.list = [1, 1000]
    return nbr_type, n_nbrs


def _metrics(truth, pred):
    resid = truth - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "R2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "RMSE": float(np.sqrt(np.mean(resid ** 2))),
        "MAE": float(np.mean(np.abs(resid))),
        "Bias": float(np.mean(resid)),
    }


def _report(metrics):
    arcpy.AddMessage("")
    for key, value in metrics.items():
        arcpy.AddMessage(f"  {key:<10} {value:>10.4f}")


class Toolbox(object):
    def __init__(self):
        self.label = "Spatial Neural Network Tools"
        self.alias = "spatialnn"
        self.tools = [
            TrainSpatialNeuralNetwork,
            PredictSpatialNeuralNetwork,
            EvaluateSpatialNeuralNetwork,
        ]


class TrainSpatialNeuralNetwork(object):
    def __init__(self):
        self.label = "Train Spatial Neural Network"
        self.description = (
            "Trains a GraphSAGE regression model with optional layer "
            "normalization and writes it to a model file. Neighborhoods are "
            "built from the input features and aggregated with a "
            "row-standardized mean. Use Predict Spatial Neural Network to "
            "apply the saved model to another feature class."
        )
        self.canRunInBackground = False

    def getParameterInfo(self):
        in_fc = arcpy.Parameter(
            displayName="Input Features",
            name="in_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )

        dependent = arcpy.Parameter(
            displayName="Dependent Variable",
            name="dependent_variable",
            datatype="Field",
            parameterType="Required",
            direction="Input",
        )
        dependent.parameterDependencies = [in_fc.name]
        dependent.filter.list = NUMERIC

        explanatory = arcpy.Parameter(
            displayName="Explanatory Variables",
            name="explanatory_variables",
            datatype="Field",
            parameterType="Required",
            direction="Input",
            multiValue=True,
        )
        explanatory.parameterDependencies = [in_fc.name]
        explanatory.filter.list = NUMERIC

        model_file = arcpy.Parameter(
            displayName="Output Model File",
            name="out_model_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Output",
        )
        model_file.filter.list = ["pth", "pt"]

        nbr_type, n_nbrs = _neighborhood_params()

        layers = arcpy.Parameter(
            displayName="Layer Sizes",
            name="layer_sizes",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        layers.value = "56, 32, 16"

        out_fc = arcpy.Parameter(
            displayName="Output Fitted Features",
            name="out_features",
            datatype="DEFeatureClass",
            parameterType="Optional",
            direction="Output",
        )

        layernorm = arcpy.Parameter(
            displayName="Use Layer Normalization",
            name="use_layer_normalization",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
            category="Advanced",
        )
        layernorm.value = True

        epochs = arcpy.Parameter(
            displayName="Maximum Epochs", name="epochs", datatype="GPLong",
            parameterType="Optional", direction="Input", category="Advanced",
        )
        epochs.value = 500

        lr = arcpy.Parameter(
            displayName="Learning Rate", name="learning_rate", datatype="GPDouble",
            parameterType="Optional", direction="Input", category="Advanced",
        )
        lr.value = 0.01

        val_prop = arcpy.Parameter(
            displayName="Validation Proportion", name="validation_proportion",
            datatype="GPDouble", parameterType="Optional", direction="Input",
            category="Advanced",
        )
        val_prop.value = 0.1

        patience = arcpy.Parameter(
            displayName="Early Stopping Patience", name="patience",
            datatype="GPLong", parameterType="Optional", direction="Input",
            category="Advanced",
        )
        patience.value = 20

        seed = arcpy.Parameter(
            displayName="Random Seed", name="random_seed", datatype="GPLong",
            parameterType="Optional", direction="Input", category="Advanced",
        )
        seed.value = 123

        return [
            in_fc, dependent, explanatory, model_file, nbr_type, n_nbrs,
            layers, out_fc, layernorm, epochs, lr, val_prop, patience, seed,
        ]

    def isLicensed(self):
        return True

    def updateParameters(self, params):
        p = {q.name: q for q in params}
        p["number_of_neighbors"].enabled = (
            p["neighborhood_type"].valueAsText == "K nearest neighbors"
        )
        return

    def updateMessages(self, params):
        p = {q.name: q for q in params}
        _check_polygon(p["in_features"], p["neighborhood_type"])
        sizes = p["layer_sizes"]
        if sizes.value:
            try:
                parsed = [int(s) for s in sizes.valueAsText.replace(",", " ").split()]
                if not parsed or any(s < 1 for s in parsed):
                    raise ValueError
            except ValueError:
                sizes.setErrorMessage(
                    "Enter one or more positive integers separated by commas, "
                    "for example 56, 32, 16."
                )
        vp = p["validation_proportion"]
        if vp.value is not None and not 0 < vp.value < 1:
            vp.setErrorMessage("Validation proportion must be between 0 and 1.")
        return

    def execute(self, params, messages):
        torch, nn = _import_torch()
        p = {q.name: q for q in params}

        in_fc = p["in_features"].valueAsText
        dep = p["dependent_variable"].valueAsText
        indep = [f.strip() for f in p["explanatory_variables"].valueAsText.split(";") if f.strip()]
        model_file = p["out_model_file"].valueAsText
        nbr_type = p["neighborhood_type"].valueAsText
        k = p["number_of_neighbors"].value or 15
        hidden = [int(s) for s in p["layer_sizes"].valueAsText.replace(",", " ").split()]
        out_fc = p["out_features"].valueAsText
        use_ln = p["use_layer_normalization"].value
        use_ln = True if use_ln is None else use_ln
        epochs = p["epochs"].value or 500
        lr = p["learning_rate"].value or 0.01
        val_prop = p["validation_proportion"].value or 0.1
        patience = p["patience"].value or 20
        seed = 123 if p["random_seed"].value is None else p["random_seed"].value

        if not os.path.splitext(model_file)[1]:
            model_file += ".pth"

        _warn_geographic(in_fc, nbr_type)
        torch.manual_seed(int(seed))
        rng = np.random.default_rng(int(seed))

        oids, xy, cols = _read_table(in_fc, [dep] + indep)
        y, x = cols[:, 0], cols[:, 1:]
        n = x.shape[0]

        arcpy.AddMessage(f"Building neighborhoods for {n} features")
        src, dst = _edges(in_fc, oids, xy, nbr_type, k)
        adj = _row_standardized_adj(torch, src, dst, n)

        n_val = max(1, int(np.floor(val_prop * n)))
        perm = rng.permutation(n)
        val_idx, train_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])

        x_mu = x[train_idx].mean(axis=0)
        x_sd = x[train_idx].std(axis=0, ddof=1)
        x_sd[x_sd == 0] = 1.0
        y_mu = float(y[train_idx].mean())
        y_sd = float(y[train_idx].std(ddof=1)) or 1.0

        x_t = torch.tensor((x - x_mu) / x_sd, dtype=torch.float32)
        y_t = torch.tensor((y - y_mu) / y_sd, dtype=torch.float32).view(-1, 1)
        tr = torch.tensor(train_idx, dtype=torch.int64)
        va = torch.tensor(val_idx, dtype=torch.int64)

        model = _build_model(torch, nn, x.shape[1], hidden, use_ln)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
        loss_fn = nn.MSELoss()

        arcpy.AddMessage(
            f"Training {'GraphSAGE + LayerNorm' if use_ln else 'GraphSAGE'} "
            f"with layers {hidden} on {len(train_idx)} features, "
            f"validating on {len(val_idx)}"
        )

        best_val, best_state, no_improve, last_epoch = float("inf"), None, 0, 0
        for epoch in range(1, int(epochs) + 1):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(x_t, adj)[tr], y_t[tr])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                v = float(loss_fn(model(x_t, adj)[va], y_t[va]))

            if v < best_val:
                best_val = v
                best_state = {a: b.detach().clone() for a, b in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch == 1 or epoch % 25 == 0:
                arcpy.AddMessage(
                    f"  epoch {epoch:4d}  train {float(loss):.4f}  validation {v:.4f}"
                )
            last_epoch = epoch
            if no_improve >= int(patience):
                arcpy.AddMessage(f"  stopped early at epoch {epoch}")
                break

        model.load_state_dict(best_state)
        model.eval()
        arcpy.AddMessage(f"Best validation loss {best_val:.4f} after {last_epoch} epochs")

        # Scaling constants travel with the weights; the saved model expects
        # standardized input and returns standardized output.
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hidden": hidden,
                "use_layer_normalization": bool(use_ln),
                "dependent_variable": dep,
                "explanatory_variables": indep,
                "x_mean": x_mu.tolist(),
                "x_std": x_sd.tolist(),
                "y_mean": y_mu,
                "y_std": y_sd,
                "neighborhood_type": nbr_type,
                "number_of_neighbors": int(k),
                "random_seed": int(seed),
                "epochs_trained": int(last_epoch),
                "best_validation_loss": float(best_val),
                "n_training_features": int(n),
            },
            model_file,
        )
        arcpy.AddMessage(f"Model saved to {model_file}")
        p["out_model_file"].value = model_file

        with torch.no_grad():
            fitted = model(x_t, adj).numpy().ravel() * y_sd + y_mu

        arcpy.AddMessage("In-sample fit, not a measure of predictive accuracy:")
        _report(_metrics(y, fitted))

        if out_fc:
            _write_predictions(in_fc, out_fc, oids, fitted, y)
            p["out_features"].value = out_fc
        return

    def postExecute(self, params):
        return


def _write_predictions(source_fc, out_fc, oids, pred, truth=None):
    arcpy.management.CopyFeatures(source_fc, out_fc)
    oid_field = arcpy.Describe(out_fc).OIDFieldName
    arcpy.management.AddField(out_fc, "PREDICTED", "DOUBLE")
    fields = [oid_field, "PREDICTED"]
    if truth is not None:
        arcpy.management.AddField(out_fc, "RESIDUAL", "DOUBLE")
        fields.append("RESIDUAL")
    lookup = {int(o): i for i, o in enumerate(oids)}
    with arcpy.da.UpdateCursor(out_fc, fields) as cur:
        for row in cur:
            i = lookup.get(int(row[0]))
            if i is None:
                continue
            row[1] = float(pred[i])
            if truth is not None:
                row[2] = float(truth[i] - pred[i])
            cur.updateRow(row)


class PredictSpatialNeuralNetwork(object):
    def __init__(self):
        self.label = "Predict Spatial Neural Network"
        self.description = (
            "Applies a model from Train Spatial Neural Network to a feature "
            "class. The prediction features are given their own graph, with no "
            "connections back to the features the model was trained on, so the "
            "prediction is inductive."
        )
        self.canRunInBackground = False

    def getParameterInfo(self):
        in_fc = arcpy.Parameter(
            displayName="Input Features",
            name="in_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )

        model_file = arcpy.Parameter(
            displayName="Input Model File",
            name="in_model_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Input",
        )
        model_file.filter.list = ["pth", "pt"]

        out_fc = arcpy.Parameter(
            displayName="Output Predicted Features",
            name="out_features",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output",
        )

        observed = arcpy.Parameter(
            displayName="Observed Variable",
            name="observed_variable",
            datatype="Field",
            parameterType="Optional",
            direction="Input",
        )
        observed.parameterDependencies = [in_fc.name]
        observed.filter.list = NUMERIC

        nbr_type, n_nbrs = _neighborhood_params(category="Neighborhood Override")
        nbr_type.parameterType = "Optional"
        nbr_type.value = None
        n_nbrs.value = None

        return [in_fc, model_file, out_fc, observed, nbr_type, n_nbrs]

    def isLicensed(self):
        return True

    def updateParameters(self, params):
        p = {q.name: q for q in params}
        p["number_of_neighbors"].enabled = (
            p["neighborhood_type"].valueAsText != "Contiguity edges corners"
            and p["neighborhood_type"].valueAsText != "Contiguity edges only"
        )
        return

    def updateMessages(self, params):
        p = {q.name: q for q in params}
        _check_polygon(p["in_features"], p["neighborhood_type"])
        return

    def execute(self, params, messages):
        torch, nn = _import_torch()
        p = {q.name: q for q in params}

        in_fc = p["in_features"].valueAsText
        model_file = p["in_model_file"].valueAsText
        out_fc = p["out_features"].valueAsText
        observed = p["observed_variable"].valueAsText

        bundle = torch.load(model_file, map_location="cpu", weights_only=False)

        indep = list(bundle["explanatory_variables"])
        nbr_type = p["neighborhood_type"].valueAsText or bundle["neighborhood_type"]
        k = p["number_of_neighbors"].value or bundle["number_of_neighbors"]

        present = {f.name for f in arcpy.ListFields(in_fc)}
        missing = [f for f in indep if f not in present]
        if missing:
            raise arcpy.ExecuteError(
                "The input features are missing explanatory variables the model "
                f"was trained on: {', '.join(missing)}."
            )

        arcpy.AddMessage(
            f"Model trained on {bundle['n_training_features']} features, "
            f"layers {bundle['hidden']}, "
            f"layer normalization {'on' if bundle['use_layer_normalization'] else 'off'}"
        )

        _warn_geographic(in_fc, nbr_type)
        fields = indep + ([observed] if observed else [])
        oids, xy, cols = _read_table(in_fc, fields)
        x = cols[:, : len(indep)]
        truth = cols[:, len(indep)] if observed else None
        n = x.shape[0]

        arcpy.AddMessage(f"Building an independent graph for {n} features")
        src, dst = _edges(in_fc, oids, xy, nbr_type, k)
        adj = _row_standardized_adj(torch, src, dst, n)

        x_mu = np.array(bundle["x_mean"])
        x_sd = np.array(bundle["x_std"])
        model = _build_model(
            torch, nn, len(indep), bundle["hidden"], bundle["use_layer_normalization"]
        )
        model.load_state_dict(bundle["state_dict"])
        model.eval()

        x_t = torch.tensor((x - x_mu) / x_sd, dtype=torch.float32)
        with torch.no_grad():
            pred = model(x_t, adj).numpy().ravel() * bundle["y_std"] + bundle["y_mean"]

        if truth is not None:
            _report(_metrics(truth, pred))

        _write_predictions(in_fc, out_fc, oids, pred, truth)
        p["out_features"].value = out_fc
        return

    def postExecute(self, params):
        return


class EvaluateSpatialNeuralNetwork(object):
    def __init__(self):
        self.label = "Evaluate Spatial Neural Network"
        self.description = (
            "Compares observed and predicted values and writes a table of "
            "accuracy measures. Residual Moran's I reports whether the model "
            "left spatial structure in its errors."
        )
        self.canRunInBackground = False

    def getParameterInfo(self):
        in_fc = arcpy.Parameter(
            displayName="Input Features",
            name="in_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )

        observed = arcpy.Parameter(
            displayName="Observed Variable",
            name="observed_variable",
            datatype="Field",
            parameterType="Required",
            direction="Input",
        )
        observed.parameterDependencies = [in_fc.name]
        observed.filter.list = NUMERIC

        predicted = arcpy.Parameter(
            displayName="Predicted Variable",
            name="predicted_variable",
            datatype="Field",
            parameterType="Required",
            direction="Input",
        )
        predicted.parameterDependencies = [in_fc.name]
        predicted.filter.list = NUMERIC

        out_table = arcpy.Parameter(
            displayName="Output Table",
            name="out_table",
            datatype="DETable",
            parameterType="Required",
            direction="Output",
        )

        residual_moran = arcpy.Parameter(
            displayName="Compute Residual Moran's I",
            name="compute_residual_moran",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        residual_moran.value = True

        nbr_type, n_nbrs = _neighborhood_params()
        nbr_type.parameterType = "Optional"

        return [in_fc, observed, predicted, out_table, residual_moran, nbr_type, n_nbrs]

    def isLicensed(self):
        return True

    def updateParameters(self, params):
        p = {q.name: q for q in params}
        on = bool(p["compute_residual_moran"].value)
        p["neighborhood_type"].enabled = on
        p["number_of_neighbors"].enabled = (
            on and p["neighborhood_type"].valueAsText == "K nearest neighbors"
        )
        return

    def updateMessages(self, params):
        p = {q.name: q for q in params}
        if p["compute_residual_moran"].value:
            _check_polygon(p["in_features"], p["neighborhood_type"])
        return

    def execute(self, params, messages):
        p = {q.name: q for q in params}
        in_fc = p["in_features"].valueAsText
        observed = p["observed_variable"].valueAsText
        predicted = p["predicted_variable"].valueAsText
        out_table = p["out_table"].valueAsText
        do_moran = bool(p["compute_residual_moran"].value)
        nbr_type = p["neighborhood_type"].valueAsText or NEIGHBORHOOD_TYPES[0]
        k = p["number_of_neighbors"].value or 15

        oids, xy, cols = _read_table(in_fc, [observed, predicted])
        truth, pred = cols[:, 0], cols[:, 1]
        n = truth.shape[0]

        metrics = _metrics(truth, pred)
        metrics["N"] = float(n)

        if do_moran:
            _warn_geographic(in_fc, nbr_type)
            src, dst = _edges(in_fc, oids, xy, nbr_type, k)
            resid = truth - pred
            z = resid - resid.mean()
            counts = np.bincount(src, minlength=n).astype(np.float64)
            counts[counts == 0] = 1.0
            # Row standardized weights make S0 equal n, so I reduces to z'Wz / z'z.
            num = float(np.sum(z[src] * z[dst] / counts[src]))
            den = float(np.sum(z ** 2))
            metrics["Residual_Moran_I"] = num / den if den else float("nan")

        _report(metrics)

        order = ["N", "R2", "RMSE", "MAE", "Bias", "Residual_Moran_I"]
        names = [key for key in order if key in metrics]
        arr = np.array(
            [(name, metrics[name]) for name in names],
            dtype=[("Measure", "U20"), ("Value", "f8")],
        )
        if arcpy.Exists(out_table):
            arcpy.management.Delete(out_table)
        arcpy.da.NumPyArrayToTable(arr, out_table)
        arcpy.AddMessage(f"Measures written to {out_table}")
        p["out_table"].value = out_table
        return

    def postExecute(self, params):
        return
