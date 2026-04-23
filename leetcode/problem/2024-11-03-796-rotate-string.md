---
layout: leetcode-entry
title: "796. Rotate String"
permalink: "/leetcode/problem/2024-11-03-796-rotate-string/"
leetcode_ui: true
entry_slug: "2024-11-03-796-rotate-string"
---

[796. Rotate String](https://leetcode.com/problems/rotate-string/description/) easy
[blog post](https://leetcode.com/problems/rotate-string/solutions/6001658/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03112024-796-rotate-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/gzfqK93qHsU)
[deep-dive](https://notebooklm.google.com/notebook/0855ee84-03bf-4750-8cd4-33f9f3aa629b/audio)
![1.webp](/assets/leetcode_daily_images/bb0a5580.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/789

#### Problem TLDR

Is string rotated `goal`? #easy #kmp #rolling-hash

#### Intuition

The brute force solution is accepted, so compare all splits.

Now, the possible optimizations:

1. Robin-Karp: precompute hash and roll it with arithmetics. Takes O(n + m) with a good hash (but can have n^2 worst case)

```j Robin-Karp

    // abc   (a * 31 + b) * 31 + c = a * 31^2 + b * 31 + c = hash
    // bc a  (b * 31 + c) * 31 + a =  (hash - a * 31^2) * 31 + a = hash * 31 - a * (31^3 - 1)
    // ca b  (hash - b * 31^2) * 31 + b
    // abcabc

```

2. Knuth-Morris-Pratt prefix-function (or z-function) https://cp-algorithms.com/string/prefix-function.html : precompute array with length of matches `suffix == prefix` for the goal, then scan the string (twice with a ring pointer % len) and find any matches with the goal in O(n + m)

```j Knuth-Morris-Pratt

    //   0123456
    // i ababaca   j g[i] g[j]  p[i]=jnew  match pref-suf
    // 0 *         0
    // 1  *        0 b    a     0          "ab"      -> ""
    // 2   *       0 a    a     1          "aba"     -> "a"
    // 3    *      1 b    b     2          "abab"    -> "ab"
    // 4     *     2 a    a     3          "ababa"   -> "aba"
    // 5      *    3 c    b     0          "ababac"  -> ""
    // 6       *   0 a    a     1          "ababaca" -> "a"

```

#### Approach

* let's implement all
* the bonus part is a golf solution c++

#### Complexity

- Time complexity:
$$O(s + g)$$

- Space complexity:
$$O(g)$$

#### Code

```kotlin

    fun rotateString(s: String, goal: String): Boolean {
        val p = s.fold(1) { r, _ -> r * 31 } - 1
        var h1 = s.fold(0) { r, c -> r * 31 + c.code }
        val h2 = goal.fold(0) { r, c -> r * 31 + c.code }
        return s.indices.any { i ->
            (h1 == h2 && s.drop(i) + s.take(i) == goal).also {
                h1 = h1 * 31  - s[i].code * p
        }}
    }

```
```rust

    pub fn rotate_string(s: String, goal: String) -> bool {
        if s.len() != goal.len() { return false }
        let (s, g) = (s.as_bytes(), goal.as_bytes());
        let (mut p, mut j) = (vec![0; g.len()], 0);
        for i in 1..g.len() {
            while j > 0 && g[i] != g[j] { j = p[j - 1] }
            if g[i] == g[j] { j += 1 }
            p[i] = j
        }
        j = 0; (0..s.len() * 2).any(|i| {
            while j > 0 && s[i % s.len()] != g[j] { j = p[j - 1] }
            if s[i % s.len()] == g[j] { j += 1 }
            j == g.len()
        })
    }

```
```c++

    bool rotateString(string s, string goal) {
        return size(s) == size(goal) && (s + s).find(goal) + 1;
    }

```

