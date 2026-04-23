---
layout: leetcode-entry
title: "2678. Number of Senior Citizens"
permalink: "/leetcode/problem/2024-08-01-2678-number-of-senior-citizens/"
leetcode_ui: true
entry_slug: "2024-08-01-2678-number-of-senior-citizens"
---

[2678. Number of Senior Citizens](https://leetcode.com/problems/number-of-senior-citizens/description/) easy
[blog post](https://leetcode.com/problems/number-of-senior-citizens/solutions/5566623/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01082024-2678-number-of-senior-citizens?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/sIosDHiKyKE)
![2024-08-01_08-08_1.webp](/assets/leetcode_daily_images/9318997b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/689

#### Problem TLDR

Count filtered by a substring #easy

#### Intuition

The `11th` and `12th` symbols are our target.

#### Approach

We can avoid Int parsing just by comparing symbols to `6` and `0`.

Let's use some API:
* Kotlin: count, drop, take
* Rust: string[..], parse, filter, count

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun countSeniors(details: Array<String>) =
        details.count { it.drop(11).take(2).toInt() > 60 }

```
```rust

    pub fn count_seniors(details: Vec<String>) -> i32 {
        details.iter().filter(|s|
            s[11..13].parse::<u8>().unwrap() > 60
        ).count() as _
    }

```

