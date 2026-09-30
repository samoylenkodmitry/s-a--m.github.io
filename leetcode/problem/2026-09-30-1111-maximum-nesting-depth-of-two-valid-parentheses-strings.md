---
layout: leetcode-entry
title: "1111. Maximum Nesting Depth of Two Valid Parentheses Strings"
permalink: "/leetcode/problem/2026-09-30-1111-maximum-nesting-depth-of-two-valid-parentheses-strings/"
leetcode_ui: true
entry_slug: "2026-09-30-1111-maximum-nesting-depth-of-two-valid-parentheses-strings"
---

[1111. Maximum Nesting Depth of Two Valid Parentheses Strings](https://leetcode.com/problems/maximum-nesting-depth-of-two-valid-parentheses-strings/solutions/8548242/kotlin-rust-by-samoylenkodmitry-581l/) medium
[substack](https://dmitriisamoilenko.substack.com/p/30092026-1111-maximum-nesting-depth?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/rPVRA34JlEQ)

https://dmitrysamoylenko.com/leetcode/

![30.09.2026.webp](/assets/leetcode_daily_images/30.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1498

#### Problem TLDR

Min braces depth subsequence split

#### Intuition

Greedily put brace in a group with the lower balance.

#### Approach

* another way: open brace olways has a group i%2, closed brace 1-i%2

#### Complexity

- Time complexity:
$$O(n)$$

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
    fun maxDepthAfterSplit(s: String) =
    s.indices.map { it + s[it].code and 1 }
```

