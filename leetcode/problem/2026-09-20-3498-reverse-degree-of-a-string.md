---
layout: leetcode-entry
title: "3498. Reverse Degree of a String"
permalink: "/leetcode/problem/2026-09-20-3498-reverse-degree-of-a-string/"
leetcode_ui: true
entry_slug: "2026-09-20-3498-reverse-degree-of-a-string"
---

[3498. Reverse Degree of a String](https://leetcode.com/problems/reverse-degree-of-a-string/solutions/8531112/kotlin-rust-by-samoylenkodmitry-9lnu/) easy
[substack](https://dmitriisamoilenko.substack.com/p/20092026-3498-reverse-degree-of-a?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ay2LL9nVUJE)

https://dmitrysamoylenko.com/leetcode/

![20.09.2026.webp](/assets/leetcode_daily_images/20.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1488

#### Problem TLDR

Sum position times reversed letter

#### Intuition

If we go from the tail, the `scan` would naturally calculates positions.

#### Approach

* `{`-c or 123-c

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun reverseDegree(s: String) =
    s.map{'{'-it}.reversed().scan(0,Int::plus).sum()
```
```rust
    pub fn reverse_degree(s: String) -> i32 {
        s.bytes().zip(1..).map(|(b, i)| i*(123-b as i32)).sum()
    }
```

