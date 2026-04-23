---
layout: leetcode-entry
title: "3403. Find the Lexicographically Largest String From the Box I"
permalink: "/leetcode/problem/2025-06-04-3403-find-the-lexicographically-largest-string-from-the-box-i/"
leetcode_ui: true
entry_slug: "2025-06-04-3403-find-the-lexicographically-largest-string-from-the-box-i"
---

[3403. Find the Lexicographically Largest String From the Box I](https://leetcode.com/problems/find-the-lexicographically-largest-string-from-the-box-i/description) medium
[blog post](https://leetcode.com/problems/find-the-lexicographically-largest-string-from-the-box-i/solutions/6809834/kotlin-rust-by-samoylenkodmitry-jba6/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04062025-3403-find-the-lexicographically?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/13d3yZcYsz8)
![1.webp](/assets/leetcode_daily_images/54b1591a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1009

#### Problem TLDR

Max string of every split by n #medium

#### Intuition

Observe an example, find a way to iterate over all possible strings.

```j
    // abcdefgh     n=3

    // a b cdefgh
    // a bc defgh
    // a bcd efgh
    // a bcde fgh
    // a bcdef gh
    // a bcdefg h

    // ab c defgh
    // ab cd efgh
    // ab cde fgh
    // ab cdef gh
    // ab cdefg h

    // abc d efgh
    // abc de fgh
    // abc def gh
    // abc defg h

    // abcd e fgh
    // abcd ef gh
    // abcd efg h

    // abcde f gh
    // abcde fg h

    // abcdef g h
    // a, ab, abc, abcd, abcde, abcdef   (length - (n-1))
    //  b bc bcd bcde bcdef bcdefg (length - (n-2) - i)
```

* for every position `i`
* the first `before = min(i, n - 1)` goes to friends
* and the last `after = n - 1 - before` goes to friends, then trim

#### Approach

* there is also O(1) memory solution, just don't do substring, save positions and compare
* there is also O(n) time solution, `1163. Last Substring in Lexicographical Order` (hard) - take the last substring, then trim; the trick is to jump to the next of `i or j` pointers by largest `s[i] or s[j]`

#### Complexity

- Time complexity:
$$O(n^2)$$, O(n) c++

- Space complexity:
$$O(n)$$, O(result) c++

#### Code

```kotlin

// 28ms
    fun answerString(w: String, n: Int) =
        if (n == 1) w else w.indices.maxOf { i ->
            w.slice(i..w.length - n + min(n - 1, i))
        }

```
```rust

// 3ms
    pub fn answer_string(w: String, n: i32) -> String {
        if n == 1 { w } else { let n = n as usize; (0..w.len())
        .map(|i| w[i..=w.len() - n + i.min(n - 1)].to_string()).max().unwrap() }
    }

```
```c++

// 79ms
    string answerString(string w, int n) {
        if (n == 1) return w; string res = "";
        for (int i = 0; i < size(w); ++i) {
            int before = min(i, n - 1);
            int after = n - 1 - before;
            string s = w.substr(i, size(w) - i - after);
            if (s > res) res = s;
        } return res;
    }

```
```c++

// 0ms
    string answerString(string w, int n) {
        if (n == 1) return w; string res = "";
        int i = 0, j = 1;
        while (j < size(w)) {
            int k = 0; while (j + k < size(w) && w[j + k] == w[i + k]) ++k;
            if (k == size(w)) break;
            if (w[j + k] > w[i + k]) i += k + 1, j = i + 1; else j += k + 1;
        }
        int sz = size(w) - n + 1; int sz1 = size(w) - i;
        return w.substr(i, min(sz, sz1));
    }

```

