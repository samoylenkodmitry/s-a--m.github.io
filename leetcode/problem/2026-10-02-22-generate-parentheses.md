---
layout: leetcode-entry
title: "22. Generate Parentheses"
permalink: "/leetcode/problem/2026-10-02-22-generate-parentheses/"
leetcode_ui: true
entry_slug: "2026-10-02-22-generate-parentheses"
---

[22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/solutions/8551747/kotlin-rust-by-samoylenkodmitry-2z0a/) medium
[substack](https://dmitriisamoilenko.substack.com/p/02102026-22-generate-parentheses?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/YXnG78jR1TA)

https://dmitrysamoylenko.com/leetcode/

![02.10.2026.webp](/assets/leetcode_daily_images/02.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1500

#### Problem TLDR

Generate all combinations of n braces pairs

#### Intuition

Generate 2^n and filter. Or divide and conquer with "(A)B" trick, by trying every split for A vs B.

#### Approach

* another way is a backtracking

#### Complexity

- Time complexity:
$$O(2^n)$$

- Space complexity:
$$O(2^n)$$

#### Code

```kotlin
    fun generateParenthesis(n: Int): List<String> =
    if (n < 1) listOf("") else (0..<n).flatMap { i ->
        generateParenthesis(i).flatMap { a ->
            generateParenthesis(n - 1 - i).map { "($a)$it" }}}
```
```rust
    pub fn generate_parenthesis(n: i32) -> Vec<String> {
        (0..1i32 << 2 * n)
        .filter(|&b|b.count_ones()==n as u32 && (0..2*n).all(|i| (b&(2<<i)-1).count_ones()*2>i as u32))
        .map(|b| (0..2 * n).map(|i| [')','('][(b>>i&1)as usize]).collect()).collect()
    }
```

