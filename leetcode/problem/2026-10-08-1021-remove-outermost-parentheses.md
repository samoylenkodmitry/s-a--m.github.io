---
layout: leetcode-entry
title: "1021. Remove Outermost Parentheses"
permalink: "/leetcode/problem/2026-10-08-1021-remove-outermost-parentheses/"
leetcode_ui: true
entry_slug: "2026-10-08-1021-remove-outermost-parentheses"
---

[1021. Remove Outermost Parentheses](https://leetcode.com/problems/remove-outermost-parentheses/solutions/8562374/kotlin-rust-by-samoylenkodmitry-7cjy/) easy
[substack](https://dmitriisamoilenko.substack.com/p/08102026-1021-remove-outermost-parentheses?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/o5f0pdEFtgQ)

https://dmitrysamoylenko.com/leetcode/

![08.10.2026.webp](/assets/leetcode_daily_images/08.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1506

#### Problem TLDR

Remove outer braces

#### Intuition

Compute running depth; filter out depth zero.

#### Approach

* Rust: .retain

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun removeOuterParentheses(s: String)=run {
        var d = 0
        s.filter { 0 < if (it<')') d++ else --d }
    }
```
```rust
    pub fn remove_outer_parentheses(mut s: String) -> String {
        let mut d = 0;
        s.retain(|c| { d += 81 - 2 * c as i32; d + c as i32 % 2 > 1 });s
    }
```

