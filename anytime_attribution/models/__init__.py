"""Attribution models with a frozen interface:

    model.fit(train_journeys, n_campaigns, seed=0) -> None
    model.credit(journey) -> np.ndarray, shape (n_campaigns)

`fit` is called once on warm-up (days 1-7) journeys only (Assumption A1); the fitted
model is then frozen and `credit` is applied to monitored (days 8-30) journeys.
`linear` and `time_decay` are theorem-covered for the aggregate share; `last_touch`,
`logistic_shap` and `darnn` are empirical, carry no identification guarantee, and
must be labelled as such wherever their output appears.
"""
