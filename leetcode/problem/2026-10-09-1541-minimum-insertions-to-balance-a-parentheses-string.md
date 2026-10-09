---
layout: leetcode-entry
title: "1541. Minimum Insertions to Balance a Parentheses String"
permalink: "/leetcode/problem/2026-10-09-1541-minimum-insertions-to-balance-a-parentheses-string/"
leetcode_ui: true
entry_slug: "2026-10-09-1541-minimum-insertions-to-balance-a-parentheses-string"
---

[1541. Minimum Insertions to Balance a Parentheses String](https://leetcode.com/problems/minimum-insertions-to-balance-a-parentheses-string/solutions/8563984/kotlin-rust-by-samoylenkodmitry-uyzg/) medium
[substack](https://dmitriisamoilenko.substack.com/p/09102026-1541-minimum-insertions?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/qNsY630fi3U)

https://dmitrysamoylenko.com/leetcode/

![09.10.2026.webp](/assets/leetcode_daily_images/09.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1507

#### Problem TLDR

Add braces to balance ( with ))

#### Intuition

Either: track current balance and move the pointer two positions forward; or track needs to be closed count.

#### Approach

* in the second case we should immediately add to odd to make close brace even if we meet an open brace

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
   fun minInsertions(s: String) = run {
        var r = 0
        s.sumOf { if (it == '(') (r % 2).also { r += 2 - it }
                  else if (--r < 0) { r = 1; 1 } else 0 } + r
    }
```
```rust
    pub fn min_insertions(s: String) -> i32 {
        let mut r = 0;
        s.bytes().map(|b| if b < 41 { let m = r % 2; r += 2 - m; m }
        else if r > 0 { r -= 1; 0 } else { r = 1; 1 }).sum::<i32>() + r
    }
```

