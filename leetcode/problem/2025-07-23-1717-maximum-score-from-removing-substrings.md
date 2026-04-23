---
layout: leetcode-entry
title: "1717. Maximum Score From Removing Substrings"
permalink: "/leetcode/problem/2025-07-23-1717-maximum-score-from-removing-substrings/"
leetcode_ui: true
entry_slug: "2025-07-23-1717-maximum-score-from-removing-substrings"
---

[1717. Maximum Score From Removing Substrings](https://leetcode.com/problems/maximum-score-from-removing-substrings/description/) medium
[blog post](https://leetcode.com/problems/maximum-score-from-removing-substrings/solutions/6993749/kotlin-rust-by-samoylenkodmitry-1byr/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23072025-1717-maximum-score-from?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/tH1yV7oJcYo)
![1.webp](/assets/leetcode_daily_images/aad99b47.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1058

#### Problem TLDR

Max removals ab=x, ba=y #medium #stack

#### Intuition

Problem is symmetric, reverse for x less than y.
Scan, put 'a' to stack, pop when meet 'b', mark removed.
Then scan again, but for 'ba'.

Notice, there are islands of 'b''s and 'a''s: we can do a single scan, and check 'ba' when we finish the curren island.

Now, the crazy part: notice how the patterns of 'a's and 'b's are always the same in the stack `bbbaaa`, so we actually don't have to use stack, just count.

#### Approach

* the finish line can be `s + '.'` or just do one extra calculation at the end

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 75ms
    fun maximumGain(s: String, x: Int, y: Int): Int {
        var r = 0; val st = Stack<Char>(); val st2 = Stack<Char>()
        val (a, b) = if (x < y) 'b' to 'a' else 'a' to 'b'
        val (x, y) = max(x, y) to min(x, y)
        for (c in s + '.')
            if (c == a) st += a
            else if (c == b) {
                if (st.size > 0 && st.peek() == a) { st.pop(); r += x } else st += b
            } else {
                for (c in st)
                    if (c == b) st2 += c
                    else if (c == a) {
                        if (st2.size > 0 && st2.peek() == b) { st2.pop(); r += y }
                    }
                st.clear(); st2.clear()
            }
        return r
    }

```
```kotlin

// 31ms
    fun maximumGain(s: String, x: Int, y: Int): Int {
        if (x < y) return maximumGain(s.reversed(), y, x)
        var a = 0; var b = 0; var r = 0
        for (c in s)
            if (c == 'a') ++a
            else if (c == 'b') if (a > 0) { --a; r += x } else ++b
            else { r += y * min(a, b); a = 0; b = 0 }
        return r + y * min(a, b)
    }

```
```rust

// 4ms
    pub fn maximum_gain(mut s: String, mut x: i32, mut y: i32) -> i32 {
        let s: Vec<_> = if x < y { s.bytes().rev().collect() } else { s.into_bytes() };
        let (mut a, mut b, mut r) = (0, 0, 0); (x, y) = (x.max(y), x.min(y));
        for c in s.into_iter() {
            if c == b'a' { a += 1 }
            else if c == b'b' { if a > 0 { a -= 1; r += x } else { b += 1 } }
            else { r += y * a.min(b); a = 0; b = 0 }
        } r + y * a.min(b)
    }

```
```c++

// 4ms
    int maximumGain(string s, int x, int y) {
        if (x < y) reverse(begin(s), end(s)), swap(x, y);
        int a = 0, b = 0, r = 0;
        for (char c : s)
            if (c == 'a') a++;
            else if (c == 'b') a ? (a--, r += x) : b++;
            else r += y * min(a, b), a = b = 0;
        return r + y * min(a, b);
    }

```

