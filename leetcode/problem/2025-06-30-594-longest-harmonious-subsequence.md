---
layout: leetcode-entry
title: "594. Longest Harmonious Subsequence"
permalink: "/leetcode/problem/2025-06-30-594-longest-harmonious-subsequence/"
leetcode_ui: true
entry_slug: "2025-06-30-594-longest-harmonious-subsequence"
---

[594. Longest Harmonious Subsequence](https://leetcode.com/problems/longest-harmonious-subsequence/description) easy
[blog post](https://leetcode.com/problems/longest-harmonious-subsequence/solutions/6901687/kotlin-rust-by-samoylenkodmitry-d31j/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30062025-594-longest-harmonious-subsequence?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/kPVCbgimEJg)
![1.webp](/assets/leetcode_daily_images/196cef06.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1035

#### Problem TLDR

Longest subsequence with max-min=1 #easy #counting #sort #two_pointers

#### Intuition

Calculate the frequencies. For each value `x` find count `x-1` and `x+1`.

Another was: sort, then linear scan.

#### Approach

* should be `exactly 1`, not `at most`
* we can check only the `x + 1` (because we `x - 1` would still be checked)

#### Complexity

- Time complexity:
$$O(n$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 25ms
    fun findLHS(n: IntArray): Int {
        val f = n.groupBy { it }
        return f.maxOf { (k, v) -> v.size + (f[k + 1]?.size ?: -v.size) }
    }

```

```rust

// 0ms
    pub fn find_lhs(mut n: Vec<i32>) -> i32 {
        n.sort_unstable(); n.chunk_by(|a, b| b - a < 1)
        .collect::<Vec<_>>().windows(2)
        .map(|w| if w[1][0] - w[0][0] == 1 { w[0].len() + w[1].len() } else { 0 })
        .max().unwrap_or(0) as _
    }

```
```c++

// 6ms
    int findLHS(vector<int>& n) {
        sort(begin(n), end(n)); int r = 0;
        for (int i = 1, j = 0; i < size(n); ++i) {
            while (j < i && n[i] - n[j] > 1) j++;
            if (n[i] - n[j] == 1) r = max(r, i - j + 1);
        } return r;
    }

```

