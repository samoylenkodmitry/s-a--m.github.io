---
layout: leetcode-entry
title: "3345. Smallest Divisible Digit Product I"
permalink: "/leetcode/problem/2026-08-06-3345-smallest-divisible-digit-product-i/"
leetcode_ui: true
entry_slug: "2026-08-06-3345-smallest-divisible-digit-product-i"
---

[3345. Smallest Divisible Digit Product I](https://leetcode.com/problems/smallest-divisible-digit-product-i/solutions/8444595/kotlin-rust-by-samoylenkodmitry-h500/) easy
[substack](https://dmitriisamoilenko.substack.com/p/06082026-3345-smallest-divisible?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/8dSYldvR3zo)

https://dmitrysamoylenko.com/leetcode/

![06.08.2026.webp](/assets/leetcode_daily_images/06.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1443

#### Problem TLDR

Smallest n.. digits product divisible by t

#### Intuition

Brute force n..inf.

#### Approach

* any zero in the number gives the satisfied result 0%t==0
* to go top-down from t to 1 we can remove each digit contribution by dividing by it's gcd(t,d)

#### Complexity

- Time complexity:
$$O(log^2(n))$$

- Space complexity:
$$O(log(n))$$

#### Code

```kotlin
    fun smallestNumber(n: Int, t: Int) =
    (n..n+9).find{"$it".fold(1){a,b->a*(b-'0')}%t<1}
```
```rust
    pub fn smallest_number(n: i32, t: i32) -> i32 {
       (n..n+10).find(|x|x.to_string().bytes().fold(1,|r,t|r*(t-b'0')as i32)%t<1).unwrap()
    }
```

