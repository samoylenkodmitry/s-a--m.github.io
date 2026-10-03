---
layout: leetcode-entry
title: "32. Longest Valid Parentheses"
permalink: "/leetcode/problem/2026-10-03-32-longest-valid-parentheses/"
leetcode_ui: true
entry_slug: "2026-10-03-32-longest-valid-parentheses"
---

[32. Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/solutions/8553410/kotlin-rust-by-samoylenkodmitry-l8uw/) hard
[substack](https://dmitriisamoilenko.substack.com/p/03102026-32-longest-valid-parentheses?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/xVI2gFx8nIE)

https://dmitrysamoylenko.com/leetcode/

![03.10.2026.webp](/assets/leetcode_daily_images/03.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1501

#### Problem TLDR

Longest valid substring

#### Intuition

Put into stack:
* the lengths
* or the left wall of the substring
then pop on closing braces and compare to max

#### Approach

* for the length: extra step to merge siblings together
* for the walls: if unbalanced - add current index as a wall

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun longestValidParentheses(s: String) = ArrayDeque(setOf(-1)).run{
        s.indices.maxOfOrNull { i ->
            if (s[i] == '(') add(i) else removeLast(); if (size<1) add(i)
            i - last()
        } ?: 0
    }
```
```rust
    pub fn longest_valid_parentheses(s: String) -> i32 {
        let mut v = vec![-1];
        s.bytes().zip(0..).map(|(b, i)| {
            if b == b'(' { v.push(i) } else { v.pop(); }
            if v.is_empty() { v.push(i) }; i - v.last().unwrap()
        }).max().unwrap_or(0)
    }
```

