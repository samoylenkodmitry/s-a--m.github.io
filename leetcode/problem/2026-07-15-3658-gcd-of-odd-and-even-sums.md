---
layout: leetcode-entry
title: "3658. GCD of Odd and Even Sums"
permalink: "/leetcode/problem/2026-07-15-3658-gcd-of-odd-and-even-sums/"
leetcode_ui: true
entry_slug: "2026-07-15-3658-gcd-of-odd-and-even-sums"
---

[3658. GCD of Odd and Even Sums](https://leetcode.com/problems/gcd-of-odd-and-even-sums/solutions/8398308/kotlin-rust-by-samoylenkodmitry-88sc/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15072026-3658-gcd-of-odd-and-even?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/7ksMlk5woyU)

https://dmitrysamoylenko.com/leetcode/

![15.07.2026.webp](/assets/leetcode_daily_images/15.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1421

#### Problem TLDR

Gcd of odds and evens

#### Intuition

Calculate sums, calculate gcd.
Or.. return `n`: sum of odds is n^2, sum of evens is n(n+1).

#### Approach

* remember gcd as `a/b, bab`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun gcdOfOddEvenSums(n: Int) = run{
        fun gcd(a: Int, b: Int): Int = if (b==0)a else gcd(b,a%b)
        gcd((1..2*n step 2).sum(), (2..2*n step 2).sum())
    }
```
```rust
    pub fn gcd_of_odd_even_sums(n: i32) -> i32 { n }
```

