---
layout: leetcode-entry
title: "961. N-Repeated Element in Size 2N Array"
permalink: "/leetcode/problem/2026-01-02-961-n-repeated-element-in-size-2n-array/"
leetcode_ui: true
entry_slug: "2026-01-02-961-n-repeated-element-in-size-2n-array"
---

[961. N-Repeated Element in Size 2N Array](https://leetcode.com/problems/n-repeated-element-in-size-2n-array/description/) easy
[blog post](https://leetcode.com/problems/n-repeated-element-in-size-2n-array/submissions/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02012026-961-n-repeated-element-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/3mUp18Z9T98)

![050f42ea-ddaf-4223-be5b-5384a627223e (1).webp](/assets/leetcode_daily_images/05d7889c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1223

#### Problem TLDR

Duplicate majority element #easy

#### Intuition

The easy intuition is just to count frequencies.

#### Approach

* frequency of 2 is enough
* if first element skipped then majority voting solution applicable
* another fun solution is to look at sum and uniq's sum

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 35ms
    fun repeatedNTimes(n: IntArray) =
        (n.sum() - n.toSet().sum())/(n.size/2-1)
```
```rust
// 0ms
    pub fn repeated_n_times(n: Vec<i32>) -> i32 {
        let (mut j, mut f) = (0, 0);
        n[(1..n.len()).find(|&i| {
            if f == 0 { j = i }
            f += (n[0]==n[i]||n[i]==n[j]) as i32 * 2 -1; f > 1
        }).unwrap_or(j)]
    }
```

