---
layout: leetcode-entry
title: "396. Rotate Function"
permalink: "/leetcode/problem/2026-05-01-396-rotate-function/"
leetcode_ui: true
entry_slug: "2026-05-01-396-rotate-function"
---

[396. Rotate Function](https://leetcode.com/problems/rotate-function/solutions/8122707/kotlin-rust-by-samoylenkodmitry-a7oh/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01052026-396-rotate-function?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/HYiuowdhlHs)

https://dmitrysamoylenko.com/leetcode/

![01.05.2026.webp](/assets/leetcode_daily_images/01.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1345

#### Problem TLDR

Max rolling hash

#### Intuition

```j
    // 0a 1b 2c 3d
    //    0b 1c 2d 3a
    // +3 -1 -1 -1
    // 3a-(b+c+d)
    //       0c 1d 2a 3b
    //    +3 -1 -1 -1
```
Reuse the previous hash to make a new. 0a1b2c3d converts to 0b1c2d3a=prev+3a-(b+c+d)=prev+3a-(sum-a)=prev+size*a-sum

#### Approach

* Rust has a nice way to (0..).zip(&n)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun maxRotateFunction(n: IntArray) = n.run {
        val s = sum(); var a = indices.sumOf { it*n[it] }
        maxOf { x -> a.also { a += size*x - s }}
    }
```
```rust
    pub fn max_rotate_function(n: Vec<i32>) -> i32 {
        let (s, mut f) = (0..).zip(&n).fold((0,0), |(s,f),(i,x)|(s+x,f+i*x));
        n.iter().map(|x| (f, f+=n.len()as i32*x-s).0).max().unwrap()
    }
```

