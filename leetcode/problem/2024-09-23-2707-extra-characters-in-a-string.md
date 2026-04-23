---
layout: leetcode-entry
title: "2707. Extra Characters in a String"
permalink: "/leetcode/problem/2024-09-23-2707-extra-characters-in-a-string/"
leetcode_ui: true
entry_slug: "2024-09-23-2707-extra-characters-in-a-string"
---

[2707. Extra Characters in a String](https://leetcode.com/problems/extra-characters-in-a-string/description/) hard
[blog post](https://leetcode.com/problems/extra-characters-in-a-string/solutions/5823037/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23092024-2707-extra-characters-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ELGGNT4sQgE)
![1.webp](/assets/leetcode_daily_images/3c332312.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/744

#### Problem TLDR

Min extra chars to form an `s` from `dictionary` #medium #dynamic_programming

#### Intuition

One way to do this is to scan `s` char by char until `word` is not in a dictionary. We can make a full Depth-First Search, memoizing the result for each start scan position. For quick dictionary check, we can use a HashSet or a Trie.

Another way is to compare the suffix of `s[..i]` with every word in a dictionary. (this solution is faster)

#### Approach

* let's implement both
* bottom-up solution can iterate forwards or backwards

#### Complexity
- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun minExtraChar(s: String, dictionary: Array<String>): Int {
        val set = dictionary.toSet(); val dp = mutableMapOf<Int, Int>()
        fun dfs(i: Int): Int = dp.getOrPut(i) {
            (i..<s.length).minOfOrNull { j ->
                dfs(j + 1) +
                if (s.substring(i, j + 1) in set) 0 else j - i + 1
            } ?: 0
        }
        return dfs(0)
    }

```
```rust

    pub fn min_extra_char(s: String, dictionary: Vec<String>) -> i32 {
        let mut dp = vec![0; s.len() + 1];
        for i in 1..=s.len() {
            dp[i] = 1 + dp[i - 1];
            for w in dictionary.iter() {
                if s[..i].ends_with(w) {
                    dp[i] = dp[i].min(dp[i - w.len()])
                }
            }
        }; dp[s.len()]
    }

```
```c++

    int minExtraChar(string s, vector<string>& dictionary) {
        vector<int> dp(s.length() + 1, 0);
        for (int i = 1; i <= s.length(); i++) {
            dp[i] = 1 + dp[i - 1];
            for (auto w: dictionary)
                if (i >= w.length() && s.substr(i - w.length(), w.length()) == w)
                    dp[i] = min(dp[i], dp[i - w.length()]);
        }
        return dp[s.length()];
    }

```

