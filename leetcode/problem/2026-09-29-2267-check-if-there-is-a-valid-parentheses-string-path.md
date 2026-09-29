---
layout: leetcode-entry
title: "2267. Check if There Is a Valid Parentheses String Path"
permalink: "/leetcode/problem/2026-09-29-2267-check-if-there-is-a-valid-parentheses-string-path/"
leetcode_ui: true
entry_slug: "2026-09-29-2267-check-if-there-is-a-valid-parentheses-string-path"
---

[2267. Check if There Is a Valid Parentheses String Path](https://leetcode.com/problems/check-if-there-is-a-valid-parentheses-string-path/solutions/8546431/kotlin-rust-by-samoylenkodmitry-tm4l/) hard
[substack](https://dmitriisamoilenko.substack.com/p/29092026-2267-check-if-there-is-a?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/KBZq0OKZbZg)

https://dmitrysamoylenko.com/leetcode/

![29.09.2026.webp](/assets/leetcode_daily_images/29.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1497

#### Problem TLDR

Any balanced braces-path

#### Intuition

Dp choice for each cell to go from top or from the left. Keep open braces balances. Remove negatives.

#### Approach

* instead of a set we can use the big integer or u128: on opened brace moves every bit to the left, meaning incrementing all at once

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun hasValidPath(g: Array<CharArray>) = run {
        val o=Array(102){0.toBigInteger()};o[1] = 1.toBigInteger()
        for (r in g) for (x in r.indices)
            o[x+1] = (o[x] or o[x+1]).shiftLeft(81 - r[x].code * 2)
        o[g[0].size].testBit(0)
    }
```
```rust
    pub fn has_valid_path(g: Vec<Vec<char>>) -> bool {
        let mut d = [0u128; 102]; d[1] = 1;
        for r in &g { for i in 0..r.len() {
            let m = d[i]|d[i+1]; d[i+1] = if r[i]<')' {m<<1} else {m>>1}
        }} d[g[0].len()] % 2 > 0
    }
```

