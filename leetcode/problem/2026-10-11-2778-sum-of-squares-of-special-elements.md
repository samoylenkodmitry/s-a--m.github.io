---
layout: leetcode-entry
title: "2778. Sum of Squares of Special Elements"
permalink: "/leetcode/problem/2026-10-11-2778-sum-of-squares-of-special-elements/"
leetcode_ui: true
entry_slug: "2026-10-11-2778-sum-of-squares-of-special-elements"
---

[2778. Sum of Squares of Special Elements](https://leetcode.com/problems/sum-of-squares-of-special-elements/solutions/8566993/kotlin-rust-by-samoylenkodmitry-tbl0/) easy
[substack](https://dmitriisamoilenko.substack.com/p/11102026-2778-sum-of-squares-of-special?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/X3WV6WoTwcc)

https://dmitrysamoylenko.com/leetcode/

![11.10.2026.webp](/assets/leetcode_daily_images/11.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1509

#### Problem TLDR

Sum of squares at positions divisible by size

#### Intuition

Brute force works.
Optimization: check only the first sqrt(n) positions, and for each use i and size/i

#### Approach

* this problem requires attention to what is actually asked to do

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun sumOfSquares1(n: IntArray)=
    n.indices.sumOf{(1-(n.size%(it+1)).sign)*n[it]*n[it]}
```
```rust
    pub fn sum_of_squares(n: Vec<i32>) -> i32 {
        n.iter().zip(1..).map(|(x,i)|x*x*((n.len()%i==0)as i32)).sum()
    }
```

