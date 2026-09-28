---
layout: leetcode-entry
title: "1614. Maximum Nesting Depth of the Parentheses"
permalink: "/leetcode/problem/2026-09-28-1614-maximum-nesting-depth-of-the-parentheses/"
leetcode_ui: true
entry_slug: "2026-09-28-1614-maximum-nesting-depth-of-the-parentheses"
---

[1614. Maximum Nesting Depth of the Parentheses](https://leetcode.com/problems/maximum-nesting-depth-of-the-parentheses/solutions/8544388/kotlin-rust-by-samoylenkodmitry-8l8i/) easy
[substack](https://dmitriisamoilenko.substack.com/p/28092026-1614-maximum-nesting-depth?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ED6MYwu_wcI)

https://dmitrysamoylenko.com/leetcode/

![28.09.2026.webp](/assets/leetcode_daily_images/28.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1496

#### Problem TLDR

Max nested braces depth

#### Intuition

Scan and calculate the max of a running sum.

#### Approach

* x.compareTo(y) gives -1,0,1

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun maxDepth(s: String) =
    s.scan(0) {r,c->r+(c=='(').compareTo(c==')')}.max()
```
```rust
    pub fn max_depth(s: String) -> i32 {
       s.bytes().fold((0,0),|(d,m),b|{let d=d+(b==40)as i32-(b==41)as i32;(d,m.max(d))}).1
    }
```

