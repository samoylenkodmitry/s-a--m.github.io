---
layout: leetcode-entry
title: "1422. Maximum Score After Splitting a String"
permalink: "/leetcode/problem/2025-01-01-1422-maximum-score-after-splitting-a-string/"
leetcode_ui: true
entry_slug: "2025-01-01-1422-maximum-score-after-splitting-a-string"
---

[1422. Maximum Score After Splitting a String](https://leetcode.com/problems/maximum-score-after-splitting-a-string/description/) easy
[blog post](https://leetcode.com/problems/maximum-score-after-splitting-a-string/solutions/6212951/kotlin-rust-by-samoylenkodmitry-771l/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01012025-1422-maximum-score-after?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/9fh_XRfyaso)
[deep-dive](https://notebooklm.google.com/notebook/d9ca7671-229d-4947-b72b-4c93854dd3d5/audio)
![1.webp](/assets/leetcode_daily_images/c9694ebb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/851

#### Problem TLDR

Max(left_zeros + right_ones) #easy

#### Intuition

The brute-force works: try every possible position split.
The better way is two-pass: count the `total ones`, then decrease it at every step.

The optimal solution is a single pass: notice, how the `sum = zeros + ones` changes at every move, we actually computing the `balance around the total ones`.

#### Approach

* try every solution
* how short can it be?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun maxScore(s: String): Int {
        var ones = s.last() - '0'; var b = 0
        return s.dropLast(1).maxOf {
            if (it > '0') { ones++; --b } else ++b
        } + ones
    }

```
```rust

    pub fn max_score(s: String) -> i32 {
        let (mut ones, mut b) = (0, 0);
        s.bytes().enumerate().map(|(i, c)| {
            ones += (c > b'0') as i32;
            if i < s.len() - 1 {
              b -= (c > b'0') as i32 * 2 - 1; } b
        }).max().unwrap() + ones
    }

```
```c++

    int maxScore(string s) {
        int o = s[s.size() - 1] == '1', b = 0, r = -1;
        for (int i = 0; i < s.size() - 1; ++i) {
            o += s[i] > '0';
            b -= (s[i] > '0') * 2 - 1;
            r = max(r, b);
        } return r + o;
    }

```

