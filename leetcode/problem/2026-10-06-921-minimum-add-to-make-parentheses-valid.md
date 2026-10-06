---
layout: leetcode-entry
title: "921. Minimum Add to Make Parentheses Valid"
permalink: "/leetcode/problem/2026-10-06-921-minimum-add-to-make-parentheses-valid/"
leetcode_ui: true
entry_slug: "2026-10-06-921-minimum-add-to-make-parentheses-valid"
---

[921. Minimum Add to Make Parentheses Valid](https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/solutions/8558801/kotlin-rust-by-samoylenkodmitry-rtbe/) medium
[substack](https://dmitriisamoilenko.substack.com/p/06102026-921-minimum-add-to-make?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/1hJO-2TFnrE)

https://dmitrysamoylenko.com/leetcode/

![06.10.2026.webp](/assets/leetcode_daily_images/06.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1504

#### Problem TLDR

Min insertions to balance braces

#### Intuition

Calculate count of times balance tries to go negative plus final balance

#### Approach

* shorter code: pop each () until its gone, the final length is unbalanced

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun minAddToMakeValid(s: String): Int =
        if ("()" in s) minAddToMakeValid(s.replace("()", "")) else s.length
```
```rust
    pub fn min_add_to_make_valid(mut s: String) -> i32 {
        while s.contains("()") { s = s.replace("()", "") } s.len() as _
    }
```

