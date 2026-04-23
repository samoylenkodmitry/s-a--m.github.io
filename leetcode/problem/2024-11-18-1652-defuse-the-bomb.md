---
layout: leetcode-entry
title: "1652. Defuse the Bomb"
permalink: "/leetcode/problem/2024-11-18-1652-defuse-the-bomb/"
leetcode_ui: true
entry_slug: "2024-11-18-1652-defuse-the-bomb"
---

[1652. Defuse the Bomb](https://leetcode.com/problems/defuse-the-bomb/description/) easy
[blog post](https://leetcode.com/problems/defuse-the-bomb/solutions/6057666/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18112024-1652-defuse-the-bomb?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/idQWsjWvjls)
[deep-dive](https://notebooklm.google.com/notebook/3fe9ec7a-6e5b-440a-8903-395c9ffa9277/audio)
![1.webp](/assets/leetcode_daily_images/fcc329c5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/804

#### Problem TLDR

Next `+-k` window sums #easy #sliding_window

#### Intuition

The problem size is small, do a brute force.

#### Approach

* to prevent off-by-ones use explicit branches of `k > 0`, `k < 0`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun decrypt(c: IntArray, k: Int) = IntArray(c.size) {
        (min(it, it + k)..max(it, it + k))
        .sumBy { c[(it + c.size) % c.size] } - c[it]
    }

```
```rust

    pub fn decrypt(c: Vec<i32>, k: i32) -> Vec<i32> {
        (0..c.len() as i32).map(|i|
            (i.min(i + k)..=i.max(i + k))
            .map(|j| c[(j as usize + c.len()) % c.len()])
            .sum::<i32>() - c[i as usize]).collect()
    }

```
```c++

    vector<int> decrypt(vector<int>& c, int k) {
        int sgn = k > 0 ? 1 : -1, s = 0, n = c.size(), d;
        vector<int> r(n, 0); if (k == 0) return r;
        if (k < 0) for (int i = n + k; i < n; ++i) s += c[i];
        if (k > 0) for (int i = 0; i < k; ++i) s += c[i];
        for (int i = 0; i < n; ++i) d = c[i] - c[(i + n + k) % n],
            s -= sgn * d, r[i] = k > 0 ? s : s - d;
        return r;
    }

```
