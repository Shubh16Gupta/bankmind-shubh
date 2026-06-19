# EXPLANATION.md — BankMind Cross-Sell API (Track C)

## What This Project Is About

The task was to build a system that helps bank Relationship Managers figure out which customers are likely to say yes to a term deposit offer. Instead of the bank calling everyone randomly and wasting time, my model looks at a customer's data and predicts the probability of them subscribing. Then I wrapped that model inside an API so any dashboard or app can just call it and get an answer back instantly.

I did Track C which means I had to do the full ML pipeline AND build a working API on top of it — not just a notebook, but something that actually runs in production.

---

## The Dataset

The dataset had 45,211 real customer records from a Portuguese bank. Each row is one customer with details like their age, job, account balance, whether they have a housing loan, and so on. The column I was trying to predict was `y` — did this person subscribe to a term deposit or not.

First thing I noticed when I loaded the data was that only about **11.7% of customers said yes** and the remaining 88.3% said no. That's a serious imbalance and it was the first real problem I had to think about before even touching any model.

---

## EDA — Looking at the Data Before Training

Before jumping into training I spent time doing exploratory data analysis to actually understand what I was working with. I made four specific plots:

- **Subscription rate by job type** — Students and retired customers had the highest subscription rates. Blue-collar workers had the lowest. This makes intuitive sense — retired people have savings and time, students are opening their first serious accounts.
- **Balance distribution** — Customers who subscribed generally had higher account balances (median around €1,500+). People with very low or negative balances almost never subscribed.
- **Age group analysis** — The 31–45 and 60+ age groups showed higher subscription rates. Younger customers (18–30) were less likely to commit.
- **Housing loan impact** — Customers without a housing loan were noticeably more likely to subscribe. If someone is already paying off a house, locking more money away in a term deposit is a harder sell.

Doing this before training was important because it helped me understand which features would actually matter and gave me something to cross-check against later when I looked at feature importance. The charts and the model ended up agreeing with each other, which was a good sign.

---

## Problems I Faced and How I Handled Them

### Problem 1 — The Imbalanced Data

This was the biggest issue. When almost 90% of your data is "no", the model just learns to say no to everyone because that way it still looks 88% accurate on paper. But that's completely useless for a bank — they specifically need to find the rare "yes" customers.

I handled this three ways:

1. Used **SMOTE** (Synthetic Minority Over-sampling Technique) — it creates realistic fake examples of the minority class so the model trains on a more balanced dataset
2. Used `scale_pos_weight` in CatBoost to tell the model that "yes" predictions matter more
3. Switched my main evaluation metric from accuracy to **F1-score**, which actually penalizes the model for ignoring the minority class

After applying SMOTE the model stopped being lazy and started actually learning what a "yes" customer looks like.

### Problem 2 — Choosing the Right Model

I started with Logistic Regression as a baseline. It gave me an F1 of 0.37 which was okay but it was missing a lot of actual "yes" customers. Then I switched to **CatBoost**.

The reason I picked CatBoost specifically is that it handles categorical columns like job, education, and marital status natively — I didn't have to manually one-hot encode every text column, CatBoost processes them internally. That saved a lot of preprocessing work and also meant less chance of me making encoding mistakes.

CatBoost trains 300 decision trees one after another where each tree learns from the mistakes of the previous one. The final prediction combines all 300 trees. This is why it performs much better than a single decision tree or even logistic regression. My CatBoost model ended up with an ROC-AUC of 0.88 compared to 0.76 for logistic regression.

### Problem 3 — The Duration Column (Data Leakage)

While going through the columns I noticed one called `duration` — the length of the last phone call with the customer. At first it seemed like a useful feature. Then I realized the problem — you only know how long a call lasted AFTER the call already happened. By that point you already know whether the customer said yes or no. Including it would be cheating because in real life you're trying to predict BEFORE making the call.

I removed it completely. This is called data leakage and it's one of those things that makes a model look great in testing but completely fail in production.

### Problem 4 — Building the API

This was my first time building a proper FastAPI application. The concept is straightforward but getting all the pieces working together took time. The main issue was that my model expects a pandas DataFrame as input but the API receives raw JSON. I had to convert the incoming JSON into a DataFrame before passing it to the model. Small thing but it breaks everything if you skip it.

I built three endpoints:

- `/health` — just returns whether the API is alive, Render pings this to check uptime
- `/predict` — takes customer data as JSON and returns the prediction, probability score, and top features that drove the decision
- `/explain` — calls Groq's LLM to write a plain English explanation of the prediction

The auto-generated Swagger docs at `/docs` also made testing much easier — I could test all three endpoints directly in the browser without writing any code.

### Problem 5 — The Groq Integration

The `/explain` endpoint was the most interesting part to build. I connected it to Groq's free API which runs llama-3.3-70b-versatile. I send it the customer's profile details and the prediction probability, and it writes back 2-3 sentences that an RM can actually use in a real conversation.

The tricky part was writing a good prompt. My first attempts gave back very generic responses like "this customer may or may not subscribe." I had to make the prompt much more specific — give it the exact numbers, tell it to focus on what an RM should do differently based on the result, and ask for a tone that's practical not just descriptive.

Once I got the prompt right, the explanations became genuinely useful. Something like: *"This 60-year-old retired customer with €2,500 in balance and no existing loans is a strong candidate — the RM should focus the conversation around financial security and fixed returns."* That's something you can actually walk into a meeting with.

### Problem 6 — Deployment on Render

I deployed everything on Render's free tier. The deployment itself wasn't too hard — connect the GitHub repo, set the build command and start command, done. But there were a few things that caught me off guard:

- Render's free tier **sleeps after inactivity**, so the first request after a while takes 30-50 seconds to respond while it wakes up. After that it's fast.
- The `GROQ_API_KEY` had to be set as an environment variable in Render's dashboard separately — on my local machine it was in a `.env` file which doesn't get committed to GitHub (and shouldn't). I forgot about this at first and the `/explain` endpoint was returning errors until I added it.
- The `--host 0.0.0.0 --port $PORT` flags in the start command are important — without them Render can't find the server.

---

## Results

The final model (CatBoost) achieved:
- **82% accuracy** on the test set
- **63% recall** — catching nearly two-thirds of actual subscribers
- **0.88 ROC-AUC** — excellent discriminative power

The model performs best on customers with:
- High account balance (> €1,500)
- No existing loans
- Age 45+
- Contact via cellular

The Logistic Regression baseline achieved 76% accuracy with lower recall (62%), confirming CatBoost was the right choice.

### Model Performance Numbers

**CatBoost (Main Model)**

| Metric | Score |
|--------|-------|
| Accuracy | 0.82 (82%) |
| F1-Score | 0.45 |
| Precision | 0.35 |
| Recall | 0.63 |
| ROC-AUC | 0.88 |

**Logistic Regression (Baseline)**

| Metric | Score |
|--------|-------|
| Accuracy | 0.76 (76%) |
| F1-Score | 0.37 |
| Precision | 0.27 |
| Recall | 0.62 |
| ROC-AUC | 0.76 |

The gap in ROC-AUC (0.88 vs 0.76) is the most telling number — it means CatBoost is significantly better at separating "yes" customers from "no" customers across all probability thresholds, not just at one fixed cutoff point.

---

## The Required Questions

**What percentage of customers have y = yes? What does this imbalance mean?**

11.7% said yes, 88.3% said no. The imbalance means a model that just predicts "no" every single time would still be 88% accurate — which sounds good but is completely useless. You can't evaluate this kind of model with accuracy alone. I used F1-score which balances precision and recall and actually tells you whether the model is doing something meaningful.

**Which job category had the highest subscription rate?**

Students and retired customers. It makes sense — retired people have money saved up and are actively looking for safe ways to grow it. Students might be setting up their first proper savings. Neither group is under the same financial pressure as someone managing a mortgage and a family.

**Which feature had the highest importance in your model?**

Balance came out on top with an importance score of 0.45. That makes complete sense — someone with a healthy account balance actually has money to put into a term deposit. Someone with a near-zero or negative balance isn't going to lock their money away for a year regardless of how good the offer is.

### Feature Importance (Top 5 from my model)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **balance** | 0.4521 |
| 2 | **pdays** | 0.2134 |
| 3 | **housing** | 0.1238 |
| 4 | **age** | 0.0892 |
| 5 | **campaign** | 0.0671 |

`pdays` being second makes sense too — if someone was contacted recently in a previous campaign, they're already warm to the idea. `housing` being third confirms what the EDA showed — people with a housing loan are harder to convert.

**Why is F1 better than accuracy here?**

Because the dataset is so lopsided. If I measured accuracy, my model could predict "no" for everyone and score 88% — looking great while being completely useless. F1 balances precision (of all the "yes" predictions I made, how many were actually correct?) and recall (of all the real "yes" customers, how many did I actually catch?). That balance is what matters when you're trying to find a rare positive class.

**Walk through one sample prediction:**

I picked a 60-year-old retired customer with €2,500 balance, no housing loan, no personal loan, contacted via cellular. The model predicted "yes" with 83.3% probability. I completely agree with this call. This person has savings, no debt burden, and fits the exact profile of someone open to locking money away safely. The balance is meaningful, the lack of loans means they have financial room, and being retired means they're probably thinking about how to make their savings work harder. I'd be surprised if this customer said no.

**What would break first if 200 RMs hit /predict simultaneously?**

The server. Right now I'm running a single instance on Render's free tier with no load balancing and no request queue. The model itself is fast but 200 simultaneous requests hitting one server would pile up and start timing out. The first thing I'd change is running multiple uvicorn workers using Gunicorn so the server can handle several requests in parallel instead of one at a time.

**What does the LLM explanation add over just showing a probability score?**

A number like "67% probability" means nothing to most Relationship Managers who aren't technical. They need to know *why* — is it because of the balance? The age? The lack of existing loans? The Groq explanation turns the number into something like "This customer has a stable financial profile with no debt commitments — approach the conversation around security and guaranteed returns." That's actionable. The probability alone just tells you what the model thinks. The explanation tells you what to actually do with that information.

---

## What I Actually Learned

Before this project I had trained models in Jupyter notebooks but never thought seriously about what happens after. Building the API forced me to think about things I'd never considered — what format does data come in, how do I validate it, what if someone sends a wrong field type, what happens under load.

The data leakage issue with the `duration` column was a good lesson. It's easy to miss things like that when you're just focused on getting a good accuracy number. You have to actually think about whether the model could work in production, not just on a test set.

SMOTE was something I hadn't used before and it genuinely made a difference to the recall on the "yes" class. And the Groq integration was surprisingly easy once I figured out the right prompt — it showed me how well a traditional ML model and an LLM complement each other. The model gives you a reliable, fast, explainable number. The LLM makes that number human.

---

## If I Had More Time

- Add **multiple workers** with Gunicorn so the API can handle concurrent requests properly
- Add **caching** so repeated predictions for the same customer don't re-run the model
- Add **SHAP values** to the `/explain` endpoint so the LLM gets more precise feature-level context to work with
- Add **input validation** with better error messages when someone sends incomplete or malformed data
- Move off the free tier so the API doesn't sleep between requests

---

## Final Thought

This project felt like a real thing — not just a notebook exercise. I was dealing with actual messy imbalanced data, catching data leakage issues, figuring out why my model was performing badly and fixing it, building an API that actually runs in production, and making predictions understandable to non-technical people through an LLM layer. That full end-to-end experience — from raw CSV to a live URL someone can actually hit — is what made it worth doing properly.