---
layout: leetcode-entry
title: "1701. Average Waiting Time"
permalink: "/leetcode/problem/2024-07-09-1701-average-waiting-time/"
leetcode_ui: true
entry_slug: "2024-07-09-1701-average-waiting-time"
---

[1701. Average Waiting Time](https://leetcode.com/problems/average-waiting-time/description/) medium
[blog post](https://leetcode.com/problems/average-waiting-time/solutions/5444506/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/9072024-1701-average-waiting-time?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/s5KyVW1BZXI)
![2024-07-09_07-43_1.webp](/assets/leetcode_daily_images/df5d5324.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/664

#### Problem TLDR

Average of intersecting intervals #medium #simulation

#### Intuition

Just simulate the process.

#### Approach

Let's use iterators to save lines of code.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun averageWaitingTime(customers: Array<IntArray>): Double {
        var time = 0
        return customers.sumOf { (start, delta) ->
            time = max(start, time) + delta
            (time - start).toDouble()
        } / customers.size
    }

```
```rust

    pub fn average_waiting_time(customers: Vec<Vec<i32>>) -> f64 {
        let mut time = 0;
        customers.iter().map(|c| {
            time = time.max(c[0]) + c[1];
            (time - c[0]) as f64
        }).sum::<f64>() / customers.len() as f64
    }

```

