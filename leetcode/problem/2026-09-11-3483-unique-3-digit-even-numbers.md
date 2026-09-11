---
layout: leetcode-entry
title: "3483. Unique 3-Digit Even Numbers"
permalink: "/leetcode/problem/2026-09-11-3483-unique-3-digit-even-numbers/"
leetcode_ui: true
entry_slug: "2026-09-11-3483-unique-3-digit-even-numbers"
---

[3483. Unique 3-Digit Even Numbers](https://leetcode.com/problems/unique-3-digit-even-numbers/solutions/8515532/kotlin-rust-by-samoylenkodmitry-xoxx/) easy
[substack](https://dmitriisamoilenko.substack.com/p/11092026-3483-unique-3-digit-even?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/cjN9zF80-UU)

https://dmitrysamoylenko.com/leetcode/

![11.09.2026.webp](/assets/leetcode_daily_images/11.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1479

#### Problem TLDR

3-digit evens consisting of given digits

#### Intuition

Brute-force either range 100..999 or all the 720 permutations of 3 out of 10

#### Approach

* Kotlin: remove does remove a single instance and returns true/false
* Rust: itertools has permutations

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun totalNumbers(d: IntArray) =
    (50..499).count{"${it*2}".map{it-'0'}.all(d.toMutableList()::remove)}
```
```rust
    pub fn total_numbers(d: Vec<i32>) -> i32 {
        d.iter().permutations(3).filter(|v| v[0] > &0 && v[2] % 2 < 1).unique().count() as _
    }
```

