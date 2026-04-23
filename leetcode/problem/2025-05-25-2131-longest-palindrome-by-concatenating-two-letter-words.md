---
layout: leetcode-entry
title: "2131. Longest Palindrome by Concatenating Two Letter Words"
permalink: "/leetcode/problem/2025-05-25-2131-longest-palindrome-by-concatenating-two-letter-words/"
leetcode_ui: true
entry_slug: "2025-05-25-2131-longest-palindrome-by-concatenating-two-letter-words"
---

[2131. Longest Palindrome by Concatenating Two Letter Words](https://leetcode.com/problems/longest-palindrome-by-concatenating-two-letter-words/description) medium
[blog post](https://leetcode.com/problems/longest-palindrome-by-concatenating-two-letter-words/solutions/6779350/kotlin-rust-by-samoylenkodmitry-hbeg/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25052025-2131-longest-palindrome?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/W42dOrwNXG8)
![1.webp](/assets/leetcode_daily_images/78bbe9d6.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/999

#### Problem TLDR

Max palindrome length from 2 char words #medium

#### Intuition

Calculate frequencies.
* count mirrors, take min f[ab], f[ba]
* take a single odd from twins f[aa] % 2

#### Approach

* don't forget *2
* take half of twins: f[aa] / 2
* we can do it one-pass, runtime is worse as we are doing more operations in the longest loop O(10^5)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 38ms
    fun longestPalindrome(w: Array<String>): Int {
        val f = w.groupBy { it }; var o = 0
        return 2 * f.entries.sumOf { (w, f1) ->
            if (w[0] == w[1]) { if (f1.size % 2 > 0) o = 2; 2 * (f1.size / 2)  }
            else min(f1.size, f[w.reversed()]?.size ?: 0) } + o
    }

```
```kotlin

// 20ms
    fun longestPalindrome(w: Array<String>): Int {
        val f = IntArray(676); var o = 0; var r = 0
        for (w in w) {
            val a = w[0] - 'a'; val b = w[1] - 'a'
            val w = a * 26 + b
            if (f[w] > 0) { --f[w]; r += 4 } else ++f[b * 26 + a]
            if (a == b) o += 2 * (f[w] and 1) - 1
        }
        return r + if (o > 0) 2 else 0
    }

```
```kotlin

// 7ms https://leetcode.com/problems/longest-palindrome-by-concatenating-two-letter-words/submissions/1643834111
    fun longestPalindrome(w: Array<String>): Int {
        val f = IntArray(676); var o = 0; var r = 0
        for (w in w) ++f[(w[0] - 'a') * 26 + (w[1] - 'a')]
        for (w in 0..675) if (f[w] > 0) r +=
            if (w / 26 == w % 26) { o = o or (f[w] and 1); f[w] and (-2) }
            else min(f[w], f[(w % 26) * 26 + w / 26])
        return 2 * (r + o)
    }

```
```rust

// 20ms
    pub fn longest_palindrome(words: Vec<String>) -> i32 {
        let (mut f, mut r, mut o) = ([0; 676], 0, 0);
        for w in words { let w = w.as_bytes();
            let (a, b) = ((w[0] - b'a') as usize, (w[1] - b'a') as usize);
            let w = a * 26 + b;
            if f[w] > 0 { f[w] -= 1; r += 4 } else { f[b * 26 + a] += 1 }
            if a == b { o += 2 * (f[w] & 1) - 1 }
        } r + (o > 0) as i32 * 2
    }

```
```rust

// 6ms https://leetcode.com/problems/longest-palindrome-by-concatenating-two-letter-words/submissions/1643877476
    pub fn longest_palindrome(words: Vec<String>) -> i32 {
        let (mut f, mut o) = ([0; 676], 0);
        for w in words { let w = w.as_bytes();
            let (a, b) = ((w[0] - b'a') as usize, (w[1] - b'a') as usize);
            f[a * 26 + b] += 1
        }
        (0..26).flat_map(|a| (a..26).map(move |b| (a, b, f[a * 26 + b])))
        .map(|(a, b, fw)| if a == b { o |= fw & 1; fw >> 1 }
            else { fw.min(f[b * 26 + a]) }).sum::<i32>() * 4 + o * 2
    }

```
```c++

// 1ms
    int longestPalindrome(vector<string>& words) {
        int f[676]={}, o = 0, r = 0;
        for (auto& s: words) {
            int a = s[0] - 'a', b = s[1] - 'a';
            int w = a * 26 + b;
            if (f[w]) --f[w], r += 4; else ++f[b * 26 + a];
            if (a == b) o += 2 * (f[w] & 1) - 1;
        } return r + 2 * (o > 0);
    }

```

