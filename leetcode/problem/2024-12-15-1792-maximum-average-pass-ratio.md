---
layout: leetcode-entry
title: "1792. Maximum Average Pass Ratio"
permalink: "/leetcode/problem/2024-12-15-1792-maximum-average-pass-ratio/"
leetcode_ui: true
entry_slug: "2024-12-15-1792-maximum-average-pass-ratio"
---

[1792. Maximum Average Pass Ratio](https://leetcode.com/problems/maximum-average-pass-ratio/description/) medium
[blog post](https://leetcode.com/problems/maximum-average-pass-ratio/solutions/6148543/kotlin-rust-by-samoylenkodmitry-8aod/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15122024-1792-maximum-average-pass?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Y3GsodbgXK8)
[deep-dive](https://notebooklm.google.com/notebook/8838b02a-d70f-4057-a8bd-90e92a6d7cb1/audio)
![1.webp](/assets/leetcode_daily_images/36e1b756.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/833

#### Problem TLDR

Arrange `passed` students to improve average score #medium #heap

#### Intuition

Didn't solve without a hint.
The hint: choose the most significant difference that can be made.

#### Approach

* in Rust we can't put `f64` into a heap, so convert into the big `i64` numbers

#### Complexity

- Time complexity:
$$O((n + m)log(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun maxAverageRatio(classes: Array<IntArray>, extraStudents: Int): Double {
        val scores = PriorityQueue<IntArray>(compareBy({
            it[0].toDouble() / it[1] - (it[0] + 1).toDouble() / (it[1] + 1) }))
        scores += classes
        for (s in 1..extraStudents)
          scores += scores.poll().also { it[0]++; it[1]++ }
        return scores.sumOf { it[0].toDouble() / it[1] } / classes.size
    }

```

```rust

    pub fn max_average_ratio(classes: Vec<Vec<i32>>, extra_students: i32) -> f64 {
        let d = |p: i64, t: i64| -> (i64, i64, i64) {
            (((t - p) * 10_000_000) / (t * t + t), p, t) };
        let mut h = BinaryHeap::from_iter(classes.iter().map(|c| {
          d(c[0] as i64, c[1] as i64) }));
        for _ in 0..extra_students {
            let (_, p, t) = h.pop().unwrap(); h.push(d(p + 1, t + 1)) }
        h.iter().map(|&(d, p, t)|
                     p as f64 / t as f64).sum::<f64>() / classes.len() as f64
    }

```
```c++

    double maxAverageRatio(vector<vector<int>>& classes, int extraStudents) {
        auto f = [&](double p, double t) { return (p + 1) / (t + 1) - p / t; };
        double r = 0; priority_queue<tuple<double, int, int>> q;
        for (auto x: classes) r += (double) x[0] / x[1],
            q.push({f(x[0], x[1]), x[0], x[1]});
        while (extraStudents--) {
            auto [d, p, t] = q.top(); q.pop();
            r += d; q.push({f(p + 1, t + 1), p + 1, t + 1});
        }
        return r / classes.size();
    }

```

