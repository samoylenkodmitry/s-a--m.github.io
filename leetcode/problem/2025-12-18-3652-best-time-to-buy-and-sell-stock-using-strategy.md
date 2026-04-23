---
layout: leetcode-entry
title: "3652. Best Time to Buy and Sell Stock using Strategy"
permalink: "/leetcode/problem/2025-12-18-3652-best-time-to-buy-and-sell-stock-using-strategy/"
leetcode_ui: true
entry_slug: "2025-12-18-3652-best-time-to-buy-and-sell-stock-using-strategy"
---

[3652. Best Time to Buy and Sell Stock using Strategy](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-using-strategy/description) medium
[blog post](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-using-strategy/solutions/7421530/kotlin-rust-by-samoylenkodmitry-b313/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18122025-3652-best-time-to-buy-and?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/uXZii7PyY8g)

![ab32cad1-a3a3-4a93-9514-929e96c0fb18 (1).webp](/assets/leetcode_daily_images/fc7fc335.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1208

#### Problem TLDR

Max Profit by modifying a sliding window sum #medum

#### Intuition

Use two separate sliding window sums.
The total max profit is `sum + (modified window - original window)``

```j
    // 0 1 2 i
    // 4 2 8
    //-1 0 1    k=2
    //   i

    // 9 2 9 5
    //-1 0 1 1    k=4
    //       i
```

#### Approach

* use a single variable to hold sliding window sum

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 22ms
    fun maxProfit(p: IntArray, st: IntArray, k: Int): Long {
        var sm = 0L; var so = 0L; var diff = 0L
        return p.indices.sumOf { i ->
            sm += p[i] - if (i-k/2 >= 0) p[i-k/2] else 0
            so += st[i] * p[i] - if (i-k >= 0) st[i-k] * p[i-k] else 0
            if (i+1-k >= 0) diff = max(diff, sm - so)
            1L * st[i] * p[i]
        } + diff
    }
```
```rust
// 0ms
    pub fn max_profit(pr: Vec<i32>, st: Vec<i32>, k: i32) -> i64 {
        let (mut sm, mut so, mut diff, k) = (0,0,0,k as usize);
        pr.iter().zip(&st).enumerate().map(|(i, (&p, &s))|{
            sm += (p - if i >= k/2 { pr[i-k/2] } else { 0 }) as i64;
            so += (s*p - if i >= k { st[i-k] * pr[i-k] } else { 0 }) as i64;
            if i + 1 >= k { diff = diff.max(sm - so) }
            (s * p) as i64
        }).sum::<i64>() + diff
    }
```

