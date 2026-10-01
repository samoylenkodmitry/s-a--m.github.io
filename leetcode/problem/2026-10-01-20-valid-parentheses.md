---
layout: leetcode-entry
title: "20. Valid Parentheses"
permalink: "/leetcode/problem/2026-10-01-20-valid-parentheses/"
leetcode_ui: true
entry_slug: "2026-10-01-20-valid-parentheses"
---

[20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/solutions/8550322/kotlin-rust-by-samoylenkodmitry-rqaz/) easy
[substack](https://dmitriisamoilenko.substack.com/p/01102026-20-valid-parentheses?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Xcdre6LvKG0)

https://dmitrysamoylenko.com/leetcode/

![01.10.2026.webp](/assets/leetcode_daily_images/01.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1499

#### Problem TLDR

Are braces valid

#### Intuition

Push open kind to the stack. Pop close kind if match with top of the stack.

#### Approach

* or push matching close brace to the stack

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun isValid(s: String) = ArrayDeque<Char>().run {
        s.all{if(it in "([{") add(it+1+it.code%2)
            else removeLastOrNull()==it } && isEmpty()}
```
```rust
    pub fn is_valid(s: String) -> bool {
        let mut q = vec![];
        s.bytes().all(|b|if b"([{".contains(&b){q.push(b+1+b%2);true}else{q.pop()==Some(b)})&&q.is_empty()
    }
```

