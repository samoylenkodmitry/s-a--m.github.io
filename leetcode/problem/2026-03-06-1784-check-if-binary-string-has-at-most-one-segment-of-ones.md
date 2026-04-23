---
layout: leetcode-entry
title: "1784. Check if Binary String Has at Most One Segment of Ones"
permalink: "/leetcode/problem/2026-03-06-1784-check-if-binary-string-has-at-most-one-segment-of-ones/"
leetcode_ui: true
entry_slug: "2026-03-06-1784-check-if-binary-string-has-at-most-one-segment-of-ones"
---

[1784. Check if Binary String Has at Most One Segment of Ones](https://open.substack.com/pub/dmitriisamoilenko/p/06032026-1784-check-if-binary-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) easy
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/06032026-1784-check-if-binary-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06032026-1784-check-if-binary-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/wI2Z8EUBRIQ)

![ccfe0859-5820-4dff-8521-1e41fe93539e (1).webp](/assets/leetcode_daily_images/7418ee05.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1289

#### Problem TLDR

1+0* pattern #easy

#### Intuition

1. Regex ^1+0*$

#### Approach

* or just check of 01

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 7ms
    fun checkOnesSegment(s: String) =
    "01" !in s
```
```rust
// 0ms
    pub fn check_ones_segment(s: String) -> bool {
        !s.contains("01")
    }
```

