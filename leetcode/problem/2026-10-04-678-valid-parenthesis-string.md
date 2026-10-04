---
layout: leetcode-entry
title: "678. Valid Parenthesis String"
permalink: "/leetcode/problem/2026-10-04-678-valid-parenthesis-string/"
leetcode_ui: true
entry_slug: "2026-10-04-678-valid-parenthesis-string"
---

[678. Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string/solutions/8555178/kotlin-rust-by-samoylenkodmitry-pk58/) medium
[substack](https://dmitriisamoilenko.substack.com/p/04102026-678-valid-parenthesis-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/N64Oc3Qcxy4)

https://dmitrysamoylenko.com/leetcode/

![04.10.2026.webp](/assets/leetcode_daily_images/04.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1502

#### Problem TLDR

Balance braces with wildcard

#### Intuition

* Dp solution: dfs and make a choice at a wildcards
* Forward-backward pass: treat wildcard as open brace, should check both ways
* Balance range solution: open brace do +1 to range, close do -1, wildcard widens the range

#### Approach

* lower should be clamped to zero, upper should not go less than zero

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun checkValidString(s: String): Boolean {
        var l = 0; var h = 0
        for (c in s) {
            l = maxOf(0, l + if (c == '(') 1 else -1)
            if ((if (c == ')') --h else ++h) < 0) return false
        }
        return l == 0
    }
```
```rust
    pub fn check_valid_string(s: String) -> bool {
        let (mut l, mut h) = (0, 0);
        s.bytes().all(|b| {
            l = (l + if b == b'(' { 1 } else { -1 }).max(0);
            h += if b == b')' { -1 } else { 1 }; h >= 0
        }) && l == 0
    }
```

