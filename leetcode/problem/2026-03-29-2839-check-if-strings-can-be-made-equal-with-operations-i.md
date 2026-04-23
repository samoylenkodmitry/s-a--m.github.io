---
layout: leetcode-entry
title: "2839. Check if Strings Can be Made Equal With Operations I"
permalink: "/leetcode/problem/2026-03-29-2839-check-if-strings-can-be-made-equal-with-operations-i/"
leetcode_ui: true
entry_slug: "2026-03-29-2839-check-if-strings-can-be-made-equal-with-operations-i"
---

[2839. Check if Strings Can be Made Equal With Operations I](https://open.substack.com/pub/dmitriisamoilenko/p/29032026-2839-check-if-strings-can?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) easy
[youtube](https://youtu.be/KyFAcXDvewg)

![29.03.2026.webp](/assets/leetcode_daily_images/29.03.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1312

#### Problem TLDR

Match a and b at (0|2,1|3) positions #easy

#### Intuition

How short can the code be?

#### Approach

* Kotlin: setOf, (0..1)
* Rust: bitmask

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 24ms
    fun canBeEqual(a: String, b: String) =
    (0..1).all { setOf(a[it],a[it+2]) == setOf(b[it],b[it+2]) }
```
```rust
// 0ms
    pub fn can_be_equal(a: String, b: String) -> bool {
        let f = |s: &[u8]| 1<<s[0] | 1<<s[2] | 1<<s[1]+16 | 1<<s[3]+16;
        f(a.as_bytes()) == f(b.as_bytes())
    }
```

