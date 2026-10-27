---
layout: leetcode-entry
title: "3963. Create Grid With Exactly One Path"
permalink: "/leetcode/problem/2026-10-27-3963-create-grid-with-exactly-one-path/"
leetcode_ui: true
entry_slug: "2026-10-27-3963-create-grid-with-exactly-one-path"
---

[3963. Create Grid With Exactly One Path](https://leetcode.com/problems/create-grid-with-exactly-one-path/solutions/8590986/kotlin-rust-by-samoylenkodmitry-eyd9/) easy
[substack](https://dmitriisamoilenko.substack.com/p/27102026-3963-create-grid-with-exactly?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/hkMvKDkHEGc)

https://dmitrysamoylenko.com/leetcode/

![27.10.2026.webp](/assets/leetcode_daily_images/27.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1524

#### Problem TLDR

Any maze with single path

#### Intuition

Top row and rigthmost column is free, others are walls

#### Approach

* repeat and string concatenation works the same on Rust and Kotlin

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun createGrid(m: Int, n: Int) =
    List(m){if (it<1)".".repeat(n) else "#".repeat(n-1)+"."}
```
```rust
    pub fn create_grid(m: i32, n: i32) -> Vec<String> {
        (0..m).map(|i| if i<1{ ".".repeat(n as usize)}else{"#".repeat(n as usize-1)+"."}).collect()
    }
```

