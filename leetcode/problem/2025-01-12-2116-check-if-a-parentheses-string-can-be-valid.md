---
layout: leetcode-entry
title: "2116. Check if a Parentheses String Can Be Valid"
permalink: "/leetcode/problem/2025-01-12-2116-check-if-a-parentheses-string-can-be-valid/"
leetcode_ui: true
entry_slug: "2025-01-12-2116-check-if-a-parentheses-string-can-be-valid"
---

[2116. Check if a Parentheses String Can Be Valid](https://leetcode.com/problems/check-if-a-parentheses-string-can-be-valid/description/) medium
[blog post](https://leetcode.com/problems/check-if-a-parentheses-string-can-be-valid/solutions/6268685/kotlin-rust-by-samoylenkodmitry-iirr/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12012025-2116-check-if-a-parentheses?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Tbxc8rJM7gA)
[deep-dive](https://notebooklm.google.com/notebook/692d1788-b774-4066-aedb-f198ae597c8a/audio)
![1.webp](/assets/leetcode_daily_images/20ebd235.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/862

#### Problem TLDR

Balance parenthesis with wildcards #medium

#### Intuition

Didn't solve it without a hint.

Some examples to observe the problem:

```j

    // 100000
    // (((()(

    // 000111
    // ()((()

    // 101111  f b
    // ((()))
    // *       0 1
    //  *      1 2

```

The corner cases that can't be balanced:
* odd string length
* locked unbalanced open brace (which is why we have to check the reversed order too)

The hint is: greedy solution just works, consider unlocked positions as wildcards, balance otherwise and check corner cases.

#### Approach

* separate counters `wildcards` and `balance` can just be a single `balance` variable, if `wildcards + balance >= 0`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun canBeValid(s: String, locked: String, o: Char = '('): Boolean {
        if (s.length % 2 > 0) return false; var b = 0
        for (i in s.indices)
            if (s[i] == o || locked[i] == '0') b++
            else if (--b < 0) return false
        return o == ')' || canBeValid(s.reversed(), locked.reversed(), ')')
    }

```
```rust

    pub fn can_be_valid(s: String, locked: String) -> bool {
        if s.len() % 2 > 0 { return false }
        let (mut b, mut o, s) = ([0, 0], [b'(', b')'], s.as_bytes());
        for i in 0..s.len() { for j in 0..2 {
            let i = if j > 0 { s.len() - 1 - i } else { i };
            if s[i] == o[j] || locked.as_bytes()[i] == b'0' { b[j] += 1 }
            else { b[j] -= 1; if b[j] < 0 { return false }}
        }}; true
    }

```
```c++

    bool canBeValid(string s, string locked) {
        if (s.size() % 2 > 0) return 0;
        int b[2] = {0}, o[2] = {'(', ')'};
        for (int i = 0; i < s.size(); ++i) for (int j = 0; j < 2; ++j) {
            int k = j ? s.size() - 1 - i : i;
            if (s[k] == o[j] || locked[k] == '0') ++b[j];
            else if (--b[j] < 0) return 0;
        } return 1;
    }

```

