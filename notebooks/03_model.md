# Notebook 03 — X-learner fit

## 1. Stage 1 — outcome models
>>> `mu0 = GBR().fit(X[T==0], Y[T==0])`
>>> `mu1 = GBR().fit(X[T==1], Y[T==1])`

## 2. Stage 2 — imputed effects
>>> `tau1_hat = Y[T==1] - mu0.predict(X[T==1])`
>>> `tau0_hat = mu1.predict(X[T==0]) - Y[T==0]`

## 3. Stage 3 — regress
>>> `tau1_model = GBR().fit(X[T==1], tau1_hat)`
>>> `tau0_model = GBR().fit(X[T==0], tau0_hat)`

## 4. Blend
>>> `cate(x) = g(x) * tau0_model.predict(x) + (1-g(x)) * tau1_model.predict(x)`

## 5. Persist
>>> `models.save(xl, "x_learner.joblib")`
