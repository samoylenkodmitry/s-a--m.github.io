---
layout: leetcode-entry
title: "1096. Brace Expansion II"
permalink: "/leetcode/problem/2026-09-25-1096-brace-expansion-ii/"
leetcode_ui: true
entry_slug: "2026-09-25-1096-brace-expansion-ii"
---

[1096. Brace Expansion II](https://leetcode.com/problems/brace-expansion-ii/solutions/8539359/kotlin-rust-by-samoylenkodmitry-wfd6/) hard
[substack](https://dmitriisamoilenko.substack.com/p/25092026-1096-brace-expansion-ii?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Je8lpqA9Kxw)

https://dmitrysamoylenko.com/leetcode/

![25.09.2026.webp](/assets/leetcode_daily_images/25.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1493

#### Problem TLDR

Expand sets-expression {x,y}z

#### Intuition

Parse recursively like a math expression, treating ',' as plus and '{' as a multiplication.
Another way - expand inner-most braces recursively.

#### Approach

* regex for innermost is `group no containing braces` "{[^{}]+}"
* or find the leftmost '}' and rightmost '{' before it

#### Complexity

- Time complexity:
$$O(n^2 3^n/3 log(n))$$

- Space complexity:
$$O(N)$$

#### Code

```kotlin
    fun braceExpansionII(e: String): List<String> =
        Regex("""\{([^{}]+)\}""").find(e)?.run {
            groupValues[1].split(',').flatMap {
                braceExpansionII(e.replaceRange(range, it)) }.toSet().sorted()
        } ?: listOf(e)
```
```rust
    pub fn brace_expansion_ii(e: String) -> Vec<String> {
        let Some(r) = e.find('}') else { return vec![e] };
        let l = e[..r].rfind('{').unwrap();
        e[l+1..r].split(',')
            .flat_map(|w| Self::brace_expansion_ii([&e[..l], w, &e[r+1..]].concat()))
            .sorted().dedup().collect()
    }
```

