---
layout: leetcode-entry
title: "1758. Minimum Changes To Make Alternating Binary String"
permalink: "/leetcode/problem/2026-03-05-1758-minimum-changes-to-make-alternating-binary-string/"
leetcode_ui: true
entry_slug: "2026-03-05-1758-minimum-changes-to-make-alternating-binary-string"
---

[1758. Minimum Changes To Make Alternating Binary String](https://open.substack.com/pub/dmitriisamoilenko/p/05032026-1758-minimum-changes-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) easy
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/05032026-1758-minimum-changes-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05032026-1758-minimum-changes-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UKM1YmZMmEQ)

![4823be9e-b115-47a6-8a53-ebca54caf6c0 (1).webp](/assets/leetcode_daily_images/4cf5f89c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1288

#### Problem TLDR

Min flips to make bits alterating #easy

#### Intuition

Check two targets: 01-type and 10-type.

#### Approach

* we can count a balance: |balance| = |correct-wrong|, L=C+W, L-|B|=(C+W)-(C-W)=2W, W=(L-|B|)/2
* (i + s[i]) %2 is an even-odd match

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 16ms
    fun minOperations(s: String) = s.length/2-
    abs(s.indices.sumOf { 1-2*((it+s[it].code)%2) })/2
```
```rust
// 0ms
    pub fn min_operations(s: String) -> i32 {
        s.len()as i32/2 - s.bytes().enumerate().map(|(i,b)|(1-2*((i as u8^b)&1)as i32)).sum::<i32>().abs()/2
    }
```

