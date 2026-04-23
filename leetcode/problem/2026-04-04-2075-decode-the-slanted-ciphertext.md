---
layout: leetcode-entry
title: "2075. Decode the Slanted Ciphertext"
permalink: "/leetcode/problem/2026-04-04-2075-decode-the-slanted-ciphertext/"
leetcode_ui: true
entry_slug: "2026-04-04-2075-decode-the-slanted-ciphertext"
---

[2075. Decode the Slanted Ciphertext](https://leetcode.com/problems/decode-the-slanted-ciphertext/solutions/7772277/kotlin-rust-by-samoylenkodmitry-pj7p/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04042026-2075-decode-the-slanted?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Nqt5FOkuY4A)

![04.04.2026.webp](/assets/leetcode_daily_images/04.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1318

#### Problem TLDR

Decode row-encoded string #medium

#### Intuition

Spaces at the end is not allowed for original string.
That means we can simply decode with the end condition, then trim.

#### Approach

* Kotlin: trimEnd

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 38ms
    fun decodeCiphertext(e: String, r: Int) = buildString {
        val skip = e.length/r + 1
        for (j in 0..<skip) for (i in 0..<r)
            if (j + i * skip < e.length) append(e[j + i * skip])
    }.trimEnd()
```
```rust
// 8ms
    pub fn decode_ciphertext(e: String, r: i32) -> String {
        let (s, mut ans) = (e.len() / r as usize + 1, String::new());
        for j in 0..s { for i in 0..r as usize {
            if j + i * s < e.len() {  ans.push(e.as_bytes()[j + i * s] as char)  }}}
        ans.trim_end().into()
    }
```

