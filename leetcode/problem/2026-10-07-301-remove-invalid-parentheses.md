---
layout: leetcode-entry
title: "301. Remove Invalid Parentheses"
permalink: "/leetcode/problem/2026-10-07-301-remove-invalid-parentheses/"
leetcode_ui: true
entry_slug: "2026-10-07-301-remove-invalid-parentheses"
---

[301. Remove Invalid Parentheses](https://leetcode.com/problems/remove-invalid-parentheses/solutions/8560645/kotlin-rust-by-samoylenkodmitry-upgn/) hard
[substack](https://dmitriisamoilenko.substack.com/p/07102026-301-remove-invalid-parentheses?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/yPf_3x7Hj5U)

https://dmitrysamoylenko.com/leetcode/

![07.10.2026.webp](/assets/leetcode_daily_images/07.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1505

#### Problem TLDR

Min removal to make balanced strings

#### Intuition

DFS+backtrack: try to remove/keep every brace, update max at the end.
BFS: each wave tries to remove every position from every string, stop at the first balanced string.

#### Approach

* dfs is easier to come up with by yourself

#### Complexity

- Time complexity:
$$O(2^n)$$

- Space complexity:
$$O(2^n)$$

#### Code

```kotlin
    fun removeInvalidParentheses(s: String): List<String> {
        fun valid(str: String) = str.scan(0) { b, c -> b + (c == '(').compareTo(c == ')') }
            .run { all { it >= 0 } && last() == 0 }
        return generateSequence(setOf(s)) { q ->
            q.flatMap { str -> str.indices.filter { str[it] in "()" }.map { str.removeRange(it..it) } }.toSet()
        }.first { q -> q.any(::valid) }.filter(::valid)
    }
```
```rust
    pub fn remove_invalid_parentheses(s: String) -> Vec<String> {
        let mut q = vec![s];
        loop {
            let (r, n): (Vec<_>,Vec<_>) = q.into_iter().partition(|s| 0 ==
                s.bytes().fold(0, |b,c| if b<0 {b} else {b+match c {40=>1,41=>-1,_=> 0}}));
            if r.len()>0 { return r }
            q = n.iter().flat_map(|s| (0..s.len()).filter(|&i| s.as_bytes()[i]/2==20).map(|i| s[..i].to_owned() + &s[i + 1..])).unique().collect()
        }
    }
```

