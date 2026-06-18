# EXPLANATION.md – Track C

## For everyone

### 1. What percentage of customers in your dataset have `y = yes`?
**~11.70%** of customers subscribed to the term deposit. This is heavily imbalanced, meaning accuracy is misleading – a dummy model predicting "no" would get ~88% accuracy. I used **F1-score** and **ROC-AUC** for evaluation.

### 2. Which job category had the highest subscription rate?
**`student`** had the highest subscription rate (~28%). This makes intuitive sense because students are younger, have fewer financial commitments, and are more open to trying new financial products.

---

## For Track C

### 3. Which feature had the highest importance in your tree‑based model?
**`balance`** had the highest feature importance. This is logical because customers with higher account balances have more disposable income and are more likely to invest in a term deposit. The bank's historical data strongly correlates higher balances with positive subscriptions.

### 4. Why is F1 a better metric than accuracy for this dataset?
Since only ~11.7% of customers subscribed, accuracy is heavily biased toward the majority class. F1-score balances **precision** (avoiding false positives) and **recall** (catching true positives). For this task, identifying potential subscribers (recall) while minimizing false alarms (precision) is critical, making F1 the best metric.

### 5. Pick one of your 5 sample predictions. Do you actually agree with the model's call?
I tested a **60-year-old retired customer** with a balance of €2,500, no housing loan, and no personal loan. The model predicted `will_subscribe: true` with **83.3% probability**. I agree because:
- Retirees typically have accumulated savings
- No existing loans means more financial freedom
- They're likely seeking safe investment options like term deposits
- The model's reasoning aligns with patterns in the EDA

### 6. What would likely break first if 200 RMs were hitting your `/predict` endpoint simultaneously?
The first bottleneck would be **CPU and memory** – model inference is synchronous and each request loads the model into memory. The Groq API would also hit rate limits quickly for the `/explain` endpoint. I would:
- Add **async endpoints** (`async def`) for non-blocking operations
- Implement **in-memory caching** for frequent customer profiles
- Use a **request queue** (e.g., Celery) for non-critical explanations
- Deploy behind a **load balancer** for horizontal scaling

### 7. What does the LLM explanation actually add over just showing a probability score?
A probability score is abstract – it tells the RM *what* the model thinks but not *why*. The LLM explanation:
- **Translates** raw predictions into human-readable, actionable language
- **Highlights key drivers** (e.g., high balance, no loans)
- **Suggests conversation approaches** for the RM
- **Builds trust** by making the "black box" transparent
- Makes the tool directly useful in real-world sales conversations