
m_lookup: process(freq_data)
begin
	case freq_data is
		when "00100100" => m_value <= "0011011";
		when "00100101" => m_value <= "0011100";
		when "00100110" => m_value <= "0011110";
		when "00100111" => m_value <= "0100000";
		when "00101000" => m_value <= "0100010";
		when "00101001" => m_value <= "0100100";
		when "00101010" => m_value <= "0100110";
		when "00101011" => m_value <= "0101000";
		when "00101100" => m_value <= "0101011";
		when "00101101" => m_value <= "0101101";
		when "00101110" => m_value <= "0110000";
		when "00101111" => m_value <= "0110011";
		when "00110000" => m_value <= "0110110";
		when "00110001" => m_value <= "0111001";
		when "00110010" => m_value <= "0111100";
		when "00110011" => m_value <= "1000000";
		when "00110100" => m_value <= "1000100";
		when "00110101" => m_value <= "1001000";
		when "00110110" => m_value <= "1001100";
		when "00110111" => m_value <= "1010000";
		when "00111000" => m_value <= "1010101";
		when "00111001" => m_value <= "1011010";
		when "00111010" => m_value <= "1011111";
		when "00111011" => m_value <= "1100101";
		when others => m_value <= "000000";

	end case;
end process;