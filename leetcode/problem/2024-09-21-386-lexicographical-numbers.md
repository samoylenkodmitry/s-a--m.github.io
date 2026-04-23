---
layout: leetcode-entry
title: "386. Lexicographical Numbers"
permalink: "/leetcode/problem/2024-09-21-386-lexicographical-numbers/"
leetcode_ui: true
entry_slug: "2024-09-21-386-lexicographical-numbers"
---

[386. Lexicographical Numbers](https://leetcode.com/problems/lexicographical-numbers/description/) medium
[blog post](https://leetcode.com/problems/lexicographical-numbers/solutions/5815677/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21092024-386-lexicographical-numbers?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2uTm__TVyP8)
![1.webp](/assets/leetcode_daily_images/e7bc3c04.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/742

#### Problem TLDR

Lexicographical ordered numbers `1..n` #medium #dfs

#### Intuition

The problem is, we should understand how the numbers are ordered:

```j

    // 1,10,100,101,102,103,104,105,106,107,108,109,
    //   11,110,111,112,113,114,115,116,117,118,119,12
    // 119 < 12 | 120

    // 1,
    // 10,
    // 100,
    // 1000,
    // 10000,
    // 10001,
    // 1001,1002,1003,1004,1005,1006,1007,1008,1009,
    // 101,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,
    // 102,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,
    // 103,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,
    // 104,1040,1041,1042,1043,1044,1045,1046,1047,1048,1049,
    // 105,1050,1051,1052,1053,1054,1055,1056,1057,1058,1059,
    // 106,1060,1061,1062,1063,1064,1065,1066,1067,1068,1069,
    // 107,1070,1071,1072,1073,1074,1075,1076,1077,1078,1079,
    // 108,1080,1081,1082,1083,1084,1085,1086,1087,1088,1089,
    // 109,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,
    // 11,
    // 110,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,111

```

Some pattern I noticed: take `102` and add digits `0..9` to the end of it, then repeat. This is a recursive problem.

Another solution is iterative: find out what the number is next - first increase `10` times, then go back `/=10` and increase by one, after that backtrack all trailing zeros `while x % 10 == 0 x/= 10`.

And finally, there is a `Trie` solution: add all numbers strings to Trie, then just scan it in a DFS order.

#### Approach

* Kotlin - simple DFS
* Rust - iterative
* c++ - Trie

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun lexicalOrder(n: Int) = buildList {
        fun dfs(x: Int): Unit = if (x <= n) {
            add(x)
            for (d in 0..9) dfs(x * 10 + d)
        } else Unit
        for (d in 1..9) dfs(d)
    }

```
```rust

    pub fn lexical_order(n: i32) -> Vec<i32> {
        let mut x = 0; (0..n).map(|i| {
            if x > 0 && x * 10 <= n { x *= 10 } else {
                if x + 1 > n { x /= 10 }
                x += 1;
                while x % 10 < 1 { x /= 10 }
            }; x
        }).collect()
    }

```
```c++

    vector<int> lexicalOrder(int n) {
        struct Trie { Trie *child[10]; int x; };
        Trie root = Trie();
        for (int i = 1; i <= n; i++) {
            Trie* t = &root;
            for (auto c: to_string(i)) {
                int next = c - '0';
                if (!t->child[next]) t->child[next] = new Trie();
                t = t->child[next];
            }
            t->x = i;
        }
        vector<int> res;
        std::function<void(Trie*)> dfs; dfs = [&](Trie* t) {
            if (t->x > 0) res.push_back(t->x);
            for (auto c: t->child) if (c) dfs(c);
        };
        dfs(&root);
        return res;
    }

```

