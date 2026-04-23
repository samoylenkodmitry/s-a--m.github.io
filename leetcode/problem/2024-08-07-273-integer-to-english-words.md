---
layout: leetcode-entry
title: "273. Integer to English Words"
permalink: "/leetcode/problem/2024-08-07-273-integer-to-english-words/"
leetcode_ui: true
entry_slug: "2024-08-07-273-integer-to-english-words"
---

[273. Integer to English Words](https://leetcode.com/problems/integer-to-english-words/description/) hard
[blog post](https://leetcode.com/problems/integer-to-english-words/solutions/5600868/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07082024-273-integer-to-english-words?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/bBIhV0dGIJM)
![1.webp](/assets/leetcode_daily_images/9ece81de.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/695

#### Problem TLDR

Integer to English words #hard

#### Intuition

Divide by 1000 and append suffix.

#### Approach

* use helper functions
* the result without extra spaces is much simpler to use

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    val s = listOf("", "One", "Two", "Three", "Four", "Five",
        "Six", "Seven", "Eight", "Nine", "Ten",
        "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen",
        "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty")
    val s10 = listOf("", "Ten", "Twenty", "Thirty", "Forty", "Fifty",
        "Sixty", "Seventy", "Eighty", "Ninety")
    val sx = listOf ("", " Thousand", " Million", " Billion", " Trillion")
    fun String.add(o: String) = if (this == "") o else if (o == "") this else "$this $o"
    fun numberToWords(num: Int): String {
        if (num < 1) return "Zero"; var x = num; var res = ""
        fun t(n: Int) = if (n < 20) s[n] else s10[n / 10].add(s[n % 10])
        fun h(n: Int, suf: String): String = if (n < 1) "" else
            (h(n / 100, " Hundred")).add(t(n % 100)) + suf
        for (suf in sx) { res = h(x % 1000, suf).add(res); x /= 1000 }
        return res
    }

```
```rust

    pub fn number_to_words(num: i32) -> String {
        let s = vec!["", "One", "Two", "Three", "Four", "Five",
        "Six", "Seven", "Eight", "Nine", "Ten",
        "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen",
        "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty"];
        let s10 = vec!["", "Ten", "Twenty", "Thirty", "Forty", "Fifty",
        "Sixty", "Seventy", "Eighty", "Ninety"];
        let sx = vec!["", " Thousand", " Million", " Billion", " Trillion"];
        fn add(a: &str, b: &str) -> String {
            if a.is_empty() { b.to_string() } else if b.is_empty() { a.to_string() }
            else { format!("{} {}", a, b) }}
        fn t(n: usize, s: &[&str], s10: &[&str]) -> String {
            if n < 20 { s[n].to_string() } else { add(s10[n / 10], s[n % 10]) }}
        fn h(n: usize, suf: &str, s: &[&str], s10: &[&str]) -> String {
            if n < 1 { String::new() } else {
                add(&h(n / 100, " Hundred", s, s10), &t(n % 100, s, s10)) + suf }}
        if num < 1 { return "Zero".to_string(); }; let (mut res, mut num) = (String::new(), num as usize);
        for suf in sx.iter() {
            res = add(&h(num % 1000, suf, &s, &s10), &res); num /= 1000;
        }; res
    }

```

