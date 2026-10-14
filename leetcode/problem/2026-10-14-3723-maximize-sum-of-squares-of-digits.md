---
layout: leetcode-entry
title: "3723. Maximize Sum of Squares of Digits"
permalink: "/leetcode/problem/2026-10-14-3723-maximize-sum-of-squares-of-digits/"
leetcode_ui: true
entry_slug: "2026-10-14-3723-maximize-sum-of-squares-of-digits"
---

[3723. Maximize Sum of Squares of Digits](https://leetcode.com/problems/maximize-sum-of-squares-of-digits/solutions/8571947/kotlin-rust-by-samoylenkodmitry-vk52/) medium
[substack](https://dmitriisamoilenko.substack.com/p/14102026-3723-maximize-sum-of-squares?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/CH9ZVRp0js4)

https://dmitrysamoylenko.com/leetcode/

![14.10.2026.webp](/assets/leetcode_daily_images/14.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1511

#### Problem TLDR

String of max n digits summing up to s

#### Intuition

Take all the nines. Then fit the leftovers.

#### Approach

* the actual number format is 999+{0..8}000+

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun maxSumOfSquares(n: Int, s: Int) =
        if (s>9*n) "" else ("9".repeat(s/9)+s%9).take(n).padEnd(n,'0')
```
```rust
    pub fn max_sum_of_squares(n: i32, s: i32) -> String {
       (0..n*(s<=9*n)as i32).map(|i|((s-9*i).clamp(0,9)+48)as u8 as char).collect()
    }
```

