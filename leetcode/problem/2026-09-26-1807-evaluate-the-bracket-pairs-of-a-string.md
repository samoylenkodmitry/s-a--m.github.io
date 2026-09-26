---
layout: leetcode-entry
title: "1807. Evaluate the Bracket Pairs of a String"
permalink: "/leetcode/problem/2026-09-26-1807-evaluate-the-bracket-pairs-of-a-string/"
leetcode_ui: true
entry_slug: "2026-09-26-1807-evaluate-the-bracket-pairs-of-a-string"
---

[1807. Evaluate the Bracket Pairs of a String](https://leetcode.com/problems/evaluate-the-bracket-pairs-of-a-string/solutions/8540874/kotlin-rust-by-samoylenkodmitry-20c2/) medium
[substack](https://dmitriisamoilenko.substack.com/p/26092026-1807-evaluate-the-bracket?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/AdPcsLrY6RA)

https://dmitrysamoylenko.com/leetcode/

![26.09.2026.webp](/assets/leetcode_daily_images/26.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1494

#### Problem TLDR

Replace keys in braces with values

#### Intuition

Find and replace.

#### Approach

* we can do regex or split by braces and collect even replace odd

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun evaluate(s: String, k: List<List<String>>) =
    k.associate { it[0] to it[1] }.let { m ->
        s.replace(Regex("""\((.*?)\)""")) { m[it.groupValues[1]] ?: "?" }}
```
```rust
    pub fn evaluate(s: String, k: Vec<Vec<String>>) -> String {
        let m: HashMap<_, _> = k.iter().map(|v| (&*v[0], &*v[1])).collect();
        s.split(['(', ')']).enumerate().map(|(i, p)| if i % 2 == 0 { p }
            else { *m.get(p).unwrap_or(&"?") }).collect()
    }
```

