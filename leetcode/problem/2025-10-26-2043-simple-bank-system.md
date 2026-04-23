---
layout: leetcode-entry
title: "2043. Simple Bank System"
permalink: "/leetcode/problem/2025-10-26-2043-simple-bank-system/"
leetcode_ui: true
entry_slug: "2025-10-26-2043-simple-bank-system"
---

[2043. Simple Bank System](https://leetcode.com/problems/simple-bank-system/description) medium
[blog post](https://leetcode.com/problems/simple-bank-system/solutions/7302551/kotlin-rust-by-samoylenkodmitry-08f3/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26102025-2043-simple-bank-system?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/-UxQODS35VQ)

https://assets.leetcode.com/users/images/537d3343-674d-4aca-855d-70851800629b_1761473639.0489593.webp

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1154

#### Problem TLDR

Design a bank #medium

#### Intuition

Reuse transfer = withdraw & deposit.

#### Approach

* carefull with off-by-ones

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 144ms
class Bank(val b: LongArray) {
    fun transfer(a1: Int, a2: Int, m: Long) =
        a2 <= b.size && withdraw(a1, m) && deposit(a2, m)
    fun deposit(a: Int, m: Long) =
        a <= b.size && { b[a-1] += m; true }()
    fun withdraw(a: Int, m: Long) =
        a <= b.size && b[a-1] >= m && { b[a-1] -= m; true }()
}

```
```rust
// 13ms
struct Bank(Vec<i64>); impl Bank {
    fn new(b: Vec<i64>) -> Self { Self(b) }
    fn transfer(&mut self, a1: i32, a2: i32, m: i64) -> bool
        { a2 as usize <= self.0.len() && self.withdraw(a1, m) && self.deposit(a2, m) }
    fn deposit(&mut self, a: i32, m: i64) -> bool
        { let a = a as usize - 1; a < self.0.len() && { self.0[a] += m; true }}
    fn withdraw(&mut self, a: i32, m: i64) -> bool {
        let a = a as usize - 1;
        a < self.0.len() && self.0[a] >= m && { self.0[a] -= m; true }
    }
}

```

