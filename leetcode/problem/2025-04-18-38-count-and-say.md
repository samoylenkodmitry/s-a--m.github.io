---
layout: leetcode-entry
title: "38. Count and Say"
permalink: "/leetcode/problem/2025-04-18-38-count-and-say/"
leetcode_ui: true
entry_slug: "2025-04-18-38-count-and-say"
---

[38. Count and Say](https://leetcode.com/problems/count-and-say/description/) medium
[blog post](https://leetcode.com/problems/count-and-say/solutions/6662777/kotlin-rust-by-samoylenkodmitry-jzzy/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18042025-38-count-and-say?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/O15knYC6nq4)
![1.webp](/assets/leetcode_daily_images/142bc549.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/962

#### Problem TLDR

nth run-length encoded string #medium

#### Intuition

Just simulate, the n is small.

#### Approach

* it is interesting to optimize this solution
* it is well-known sequence `A005150` named "Look-and-say" https://en.wikipedia.org/wiki/Look-and-say_sequence https://oeis.org/A005150
* the only numbers are 1, 2 and 3

#### Complexity

- Time complexity:
$$O(n^2)$$, according to wiki, it grows 30% per generation

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun countAndSay(n: Int): String = if (n < 2) "1" else buildString {
        val s = countAndSay(n - 1); var i = 0; var j = 0
        while (i < s.length) {
            while (j < s.length && s[i] == s[j]) j++
            append(j - i).append(s[i]); i = j
        }
    }

```
```kotlin

    fun countAndSay(n: Int): String {
        var a = CharArray((49 * n * n + 20 * n) / 10); var b = CharArray(a.size)
        a[0] = '1'; var sz = 1
        for (r in 2..n) {
            var i = 0; var j = 0; var k = 0
            while (i < sz) {
                val x = a[i]
                while (j < sz && a[j] == x) j++
                b[k++] = '0' + j - i
                b[k++] = x
                i = j
            }
            a = b.also { b = a }; sz = k
        }
        return String(a, 0, sz)
    }

```
```kotlin

val answers = {
    var a = CharArray(3410); var b = CharArray(4463)
    a[0] = '1'; var sz = 1; val res = Array(31) { "1" }
    for (r in 2..30) {
        var i = 0; var j = 0; var k = 0
        while (i < sz) {
            val x = a[i]
            while (j < sz && a[j] == x) j++
            b[k++] = '0' + j - i
            b[k++] = x
            i = j
        }
        a = b.also { b = a }; sz = k
        res[r] = String(a, 0, sz)
    }
    res
}()
class Solution { fun countAndSay(n: Int) = answers[n] }

```
```rust

    pub fn count_and_say(n: i32) -> String {
        if n < 2 { return "1".into() }
        let s = Self::count_and_say(n - 1); let s = s.as_bytes();
        let (mut i, mut j, mut v) = (0, 0, vec![]);
        while i < s.len() {
            while j < s.len() && s[i] == s[j] { j += 1 }
            v.push(b'0' + (j - i) as u8); v.push(s[i]); i = j
        } String::from_utf8(v).unwrap()
    }

```
```c++

    string countAndSay(int n) {
        if (n < 2) return "1";
        string s = countAndSay(n - 1), r;
        for (int i = 0, j = 0; i < size(s); i = j) {
            while (j < size(s) && s[i] == s[j]) ++j;
            r += '0' + j - i; r += s[i];
        } return r;
    }

```

