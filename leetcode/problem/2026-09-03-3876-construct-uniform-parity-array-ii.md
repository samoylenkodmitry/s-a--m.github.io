---
layout: leetcode-entry
title: "3876. Construct Uniform Parity Array II"
permalink: "/leetcode/problem/2026-09-03-3876-construct-uniform-parity-array-ii/"
leetcode_ui: true
entry_slug: "2026-09-03-3876-construct-uniform-parity-array-ii"
---

[3876. Construct Uniform Parity Array II](https://leetcode.com/problems/construct-uniform-parity-array-ii/solutions/8498940/kotlin-rust-by-samoylenkodmitry-h49a/) medium
[substack](https://dmitriisamoilenko.substack.com/p/03092026-3876-construct-uniform-parity?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EjS1QG7CxrA)

https://dmitrysamoylenko.com/leetcode/

![03.09.2026.webp](/assets/leetcode_daily_images/03.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1471

#### Problem TLDR

Can make all even/odd by subtracting any smaller

#### Intuition

Subtracting the odd converts everything to odd. We have to have the smallest odd to subtract from any other.

#### Approach

* write the most explicit code, then shrink whats irrelevant

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun uniformArray(n: IntArray) =
        n.all{it%2<1} || n.min()%2>0
```
```rust
    pub fn uniform_array(n: Vec<i32>) -> bool {
       n.iter().all(|x|x&1<1)||n.iter().min().unwrap()&1>0
    }
```

