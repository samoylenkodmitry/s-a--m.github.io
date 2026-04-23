---
layout: leetcode-entry
title: "2073. Time Needed to Buy Tickets"
permalink: "/leetcode/problem/2024-04-09-2073-time-needed-to-buy-tickets/"
leetcode_ui: true
entry_slug: "2024-04-09-2073-time-needed-to-buy-tickets"
---

[2073. Time Needed to Buy Tickets](https://leetcode.com/problems/time-needed-to-buy-tickets/description/) easy
[blog post](https://leetcode.com/problems/time-needed-to-buy-tickets/solutions/4996548/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09042024-2073-time-needed-to-buy?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/9IQCe-YO2I0)
![2024-04-09_08-27.webp](/assets/leetcode_daily_images/39fc08c1.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/565

#### Problem TLDR

Seconds to buy tickets by `k`-th person in a rotating 1 second queue #easy

#### Intuition

The brute-force implementation is trivial: just repeat decreasing `tickets[i]` untile `tickets[k] == 0`. It will take at most O(n^2) time.

However, there is a one-pass solution. To get the intuition go to the comment section... just a joke. We take `tickets[k]` for people before `k` and we don't take last round tickets for people after `k`, so only `tickets[k] - 1`.

#### Approach

Let's use some iterators to reduce the number of lines of code:
`sumOf`, `withIndex` or `iter().enumerate()`,

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun timeRequiredToBuy(tickets: IntArray, k: Int) =
        tickets.withIndex().sumOf { (i, t) ->
            min(tickets[k] - (if (i > k) 1 else 0), t)
    }

```
```rust

    pub fn time_required_to_buy(tickets: Vec<i32>, k: i32) -> i32 {
        tickets.iter().enumerate().map(|(i, &t)|
            t.min(tickets[k as usize] - i32::from(i > k as usize))).sum()
    }

```

