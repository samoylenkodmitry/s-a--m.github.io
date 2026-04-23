---
layout: leetcode-entry
title: "1716. Calculate Money in Leetcode Bank"
permalink: "/leetcode/problem/2025-10-25-1716-calculate-money-in-leetcode-bank/"
leetcode_ui: true
entry_slug: "2025-10-25-1716-calculate-money-in-leetcode-bank"
---

[1716. Calculate Money in Leetcode Bank](https://leetcode.com/problems/calculate-money-in-leetcode-bank/description/) easy
[blog post](https://leetcode.com/problems/calculate-money-in-leetcode-bank/solutions/7299247/kotlin-rust-by-samoylenkodmitry-2s5k/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25102025-1716-calculate-money-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/n30r1NZOE4w)

![4ae592a1-851f-4053-b5e8-d1d641ed2a4b (1).webp](/assets/leetcode_daily_images/65d952b2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1153

#### Problem TLDR

Add increasing sum of money, drop weekly gain on mondays #easy

#### Intuition

Simulate.

```j
    // 0 1 2 3  4  5  6
    // 7 8 9 10 11 12 13
    // 14
    // 1 2 3 4 5 6 7
    // 2 3 4 5 6 7 8   or prev + 7
    // 3 4 5 6 7 8 9   or prev + 7
    // 4 5 6 7 8 9 10  or prev + 7
```
The O(1) solutino: count weeks and remainder of days.
Each week contributes as a sum of base*7. Contribution of each bases is w*(w-1)/2.
Another week contribution is weekly gains, 7*(7+1)/2. They are just `w*gains`

#### Approach

* draw the numbers to better understand the laws

#### Complexity

- Time complexity:
$$O(n)$$ or O(1)

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 8ms
    fun totalMoney(n: Int) = (0..<n).sumOf { it % 7 + it / 7 + 1 }

```
```rust
// 0ms
    pub fn total_money(n: i32) -> i32 {
        let w = n/7; let d = n-7*w;
        let fullweeks = 28*w + 7*w*(w-1)/2;
        let taildays = d*(d+1)/2 + w*d;
        fullweeks + taildays
    }

```

