---
layout: leetcode-entry
title: "1415. The k-th Lexicographical String of All Happy Strings of Length n"
permalink: "/leetcode/problem/2025-02-19-1415-the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/"
leetcode_ui: true
entry_slug: "2025-02-19-1415-the-k-th-lexicographical-string-of-all-happy-strings-of-length-n"
---

[1415. The k-th Lexicographical String of All Happy Strings of Length n](https://leetcode.com/problems/the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/description/) medium
[blog post](https://leetcode.com/problems/the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/solutions/6441193/kotlin-rust-by-samoylenkodmitry-hhxx/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19022025-1415-the-k-th-lexicographical?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Anl601V2eQQ)
![1.webp](/assets/leetcode_daily_images/ba24ba08.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/900

#### Problem TLDR

`k`th string of `abc` permutations #medium #backtracking

#### Intuition

The brute-force is accepted and trivial: try everything in a DFS.

The math solution from u/votrubac/:

```j

    // n = 3 k = 9, k-- = 8
    // comb = 2^2 = 4, 3 * comb = 12
    // k / comb = 9 / 4 = 2 -> 'a' + 2 = 'c'
    // 'c'
    // k = k % comb = 8 % 4 = 0, p = 'c'
    // comb /= 2 = 2
    // k < comb ? 0 < 2, 'a' + (p=='a' = 0), 'ca'
    // k = k % comb = 0 % 2 = 1, p = 'a'
    // comb /= 2 = 1
    // k < comb ? 0 < 1, 'a' + (p =='a'=1), 'cab'

```

It works, but what is exactly `k %= comb, comb /= 2` doing?
For the string length of `n` there are `2 ^ n` combinations (of what?, why?).
We are shortening the string by `1`. Each new subproblem is a choice between starting sets of `ab` vs `bc`. If `k < comb` we are in `ab` territory, otherwise `bc`. (how so? idk). Next, there is a help of checking the previous letter to choose between the two `a` or `b` from `ab` and `b` or `c` from `bc`. I think I'm failing to grok the intuition behind this, so let's postpone it for the next time.

#### Approach

* write DFS

#### Complexity

- Time complexity:
$$O(nk)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun getHappyString(n: Int, k: Int): String {
        var strs = 0
        fun dfs(soFar: String): String? =
            if (soFar.length == n) { if (++strs == k) soFar else null }
            else listOf('a', 'b', 'c').firstNotNullOfOrNull { c ->
                if (soFar.lastOrNull() != c) dfs(soFar + c) else null }
        return dfs("") ?: ""
    }

```
```rust

    pub fn get_happy_string(n: i32, mut k: i32) -> String {
        let mut comb = 1 << (n - 1); if k > 3 * comb { return "".into() }
        k -= 1; let mut res = vec![b'a' + (k / comb) as u8];
        while comb > 1 {
            k %= comb; comb /= 2; let p = res[res.len() - 1];
            res.push(if k < comb { b'a' + (p == b'a') as u8 } else { b'c' - (p == b'c') as u8 });
        }; String::from_utf8(res).unwrap()
    }

```
```c++

    string getHappyString(int n, int k) {
        int comb = 1 << (n - 1); if (k > 3 * comb) return "";
        k--; string res(1, 'a' + k / comb);
        while (comb > 1)
            k %= comb, comb /= 2,
            res += k < comb ? 'a' + (res.back() == 'a') : 'c' - (res.back() == 'c');
        return res;
    }

```

