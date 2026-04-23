---
layout: leetcode-entry
title: "3637. Trionic Array I"
permalink: "/leetcode/problem/2026-02-03-3637-trionic-array-i/"
leetcode_ui: true
entry_slug: "2026-02-03-3637-trionic-array-i"
---

[3637. Trionic Array I](https://leetcode.com/problems/trionic-array-i/) easy
[blog post](https://leetcode.com/problems/trionic-array-i/solutions/7548258/kotlin-rust-by-samoylenkodmitry-2itq/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03022026-3637-trionic-array-i?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/POsTm9HyUQk)

![6836c4d7-90a4-489e-b6b1-2efbe8b6a980 (1).webp](/assets/leetcode_daily_images/25101d2b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1257

#### Problem TLDR

Validate inc,dec,inc sequency #easy

#### Intuition

Brute-Force O(n^3) is accepted.

#### Approach

* we can count peaks
* we can convert to string and use regex
* chunk_by, eq in Rust

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 32ms
    fun isTrionic(n: IntArray) =
    n.map{it}.zipWithNext(Int::compareTo)
        .joinToString("").matches(Regex("(-1)+1+(-1)+"))
    /*
    n.map{it}.zipWithNext(Int::compareTo).let { s ->
        s[0]<0 && 0 !in s && s.windowed(2).count{(a,b)->a!=b}==2 }

    n[0]<n[1] && 2 == (2..<n.size).count {
       (n[it-2]<n[it-1])!=(n[it-1]<n[it]) } &&
       (1..<n.size).all{n[it]!=n[it-1]}
    */
```
```rust
// 0ms
    pub fn is_trionic(n: Vec<i32>) -> bool {
        n.windows(2).map(|w|w[1].cmp(&w[0])as i8).collect::<Vec<_>>()
        .chunk_by(i8::eq).map(|c|c[0]).eq([1,-1,1])
    }
```

